"""
FastAPI Backend Application for AI-Based Crop Health Monitoring System.

Endpoints:
- /api/health : System status and loaded model metadata
- /api/classes : Supported crop disease categories
- /api/predict/image : Leaf image classification & Grad-CAM visual explanation
- /api/environmental/analyze : Random Forest microclimate risk analytics
- /api/crop-health/assess : Multi-modal fusion returning Unified Crop Health Index & Recommendations
- /api/history : Read, write, and manage diagnosis history records
"""

import os
import sys
import io
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from src.predict import get_predictor
from src.environmental_analysis import (
    analyze_environmental_conditions,
    compute_unified_crop_health,
    load_environmental_model
)

HISTORY_FILE = PROJECT_ROOT / "outputs" / "prediction_history.json"


app = FastAPI(
    title="AI-Based Crop Health Monitoring API",
    description="Deep learning plant disease diagnosis fused with microclimate environmental risk analytics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class EnvironmentalInput(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius (e.g. 24.5)", ge=-10.0, le=60.0)
    humidity: float = Field(..., description="Relative humidity percentage (e.g. 75.0)", ge=0.0, le=100.0)
    soil_moisture: float = Field(..., description="Soil moisture percentage (e.g. 60.0)", ge=0.0, le=100.0)
    data_source: str = Field(default="simulated", description="'actual_sensor' or 'simulated'")


class AssessmentRequest(BaseModel):
    prediction: Dict[str, Any]
    environmental: Dict[str, Any]


def load_history() -> List[Dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(records: List[Dict[str, Any]]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(records, f, indent=2)


@app.get("/api/health")
def health_check():
    predictor = get_predictor()
    return {
        "status": "healthy",
        "service": "AI-Based Crop Health Monitoring System",
        "device": str(predictor.device),
        "supported_classes_count": len(predictor.classes),
        "active_models": ["Custom CNN", "MobileNetV2 (Transfer Learning)", "Random Forest (Environmental)"]
    }


@app.get("/api/classes")
def get_classes():
    predictor = get_predictor()
    return {
        "classes": predictor.classes,
        "class_details": [
            {
                "id": c,
                "display_name": c.replace("___", " - ").replace("_", " "),
                "type": "Healthy Reference" if "healthy" in c.lower() else "Pathogen / Disease"
            }
            for c in predictor.classes
        ]
    }


@app.post("/api/predict/image")
async def predict_image_endpoint(
    file: UploadFile = File(...),
    model_type: str = Form(default="mobilenet"),
    generate_gradcam: bool = Form(default=True)
):
    """
    Classifies uploaded leaf image and returns diagnosis + base64 Grad-CAM overlay.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    predictor = get_predictor()
    result = predictor.predict(pil_img, model_type=model_type, generate_cam=generate_gradcam)

    response = {
        "predicted_class": result["predicted_class"],
        "display_name": result["display_name"],
        "confidence": result["confidence"],
        "is_healthy": result["is_healthy"],
        "model_type": result["model_type"],
        "all_probabilities": result["all_probabilities"],
    }

    if generate_gradcam and "gradcam" in result:
        overlay_buf = io.BytesIO()
        result["gradcam"]["overlay_image"].save(overlay_buf, format="PNG")
        overlay_b64 = base64.b64encode(overlay_buf.getvalue()).decode("utf-8")

        heatmap_buf = io.BytesIO()
        result["gradcam"]["heatmap_image"].save(heatmap_buf, format="PNG")
        heatmap_b64 = base64.b64encode(heatmap_buf.getvalue()).decode("utf-8")

        response["gradcam"] = {
            "overlay_base64": overlay_b64,
            "heatmap_base64": heatmap_b64
        }

    return response


@app.post("/api/environmental/analyze")
def analyze_environmental_endpoint(env_input: EnvironmentalInput):
    """
    Evaluates microclimate parameters using Random Forest model.
    """
    result = analyze_environmental_conditions(
        temperature=env_input.temperature,
        humidity=env_input.humidity,
        soil_moisture=env_input.soil_moisture,
        data_source=env_input.data_source
    )
    return result


@app.post("/api/crop-health/assess")
def assess_crop_health_endpoint(data: AssessmentRequest):
    """
    Fuses vision predictions and environmental analytics into unified Crop Health Index.
    """
    unified = compute_unified_crop_health(data.prediction, data.environmental)

    # Persist into history
    record = {
        "timestamp": datetime.now().isoformat(),
        "crop_health_index": unified["crop_health_index"],
        "overall_status": unified["overall_status"],
        "predicted_class": data.prediction.get("display_name", data.prediction.get("predicted_class")),
        "confidence": data.prediction.get("confidence"),
        "is_healthy": unified["is_healthy_detected"],
        "temperature": data.environmental.get("temperature"),
        "humidity": data.environmental.get("humidity"),
        "soil_moisture": data.environmental.get("soil_moisture"),
        "environmental_risk_level": data.environmental.get("risk_level"),
        "data_source": data.environmental.get("data_source", "simulated"),
        "recommendations": unified["recommendations"]
    }

    records = load_history()
    records.insert(0, record)
    # Cap history at 100 entries
    save_history(records[:100])

    return unified


@app.get("/api/history")
def get_prediction_history(limit: int = Query(default=20, ge=1, le=100)):
    records = load_history()
    return {
        "total_records": len(records),
        "history": records[:limit]
    }


@app.delete("/api/history")
def clear_prediction_history():
    save_history([])
    return {"message": "Prediction history successfully cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend:app", host="127.0.0.1", port=8000, reload=False)
