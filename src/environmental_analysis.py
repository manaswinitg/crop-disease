"""
Environmental Data Analytics & Multi-Modal Crop Health Fusion Module.

Analyzes microclimate variables (Temperature, Relative Humidity, Soil Moisture)
using a trained Random Forest model to compute Environmental Disease Vulnerability.
Combines visual deep learning inferences with environmental risk into a unified Crop Health Index.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "environmental_rf.joblib")


def calculate_vpd(temperature_c: float, humidity_percent: float) -> float:
    """
    Computes Vapor Pressure Deficit (VPD in kPa).
    VPD is a vital physiological driver of transpiration and fungal spore germination.
    SVP = 0.61078 * exp((17.27 * T) / (T + 237.3))
    VPD = SVP * (1 - RH / 100)
    """
    svp = 0.61078 * math.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    vpd = svp * (1.0 - (humidity_percent / 100.0))
    return max(0.0, vpd)


def generate_agronomic_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generates an agronomic microclimate dataset based on empirical plant pathology rules.
    Features:
      - temperature (°C): 10.0 to 42.0
      - humidity (%): 20.0 to 98.0
      - soil_moisture (%): 10.0 to 95.0
    Labels:
      0: Low Risk (Optimal)
      1: Moderate Risk (Suboptimal conditions / Mild stress)
      2: High Risk (High fungal/bacterial proliferation potential)
      3: Extreme Risk (Environmental shock / Acute failure)
    """
    np.random.seed(seed)

    temps = np.random.uniform(10.0, 42.0, n_samples)
    humids = np.random.uniform(20.0, 98.0, n_samples)
    moists = np.random.uniform(10.0, 95.0, n_samples)

    labels = []
    vpds = []

    for t, h, m in zip(temps, humids, moists):
        vpd = calculate_vpd(t, h)
        vpds.append(vpd)

        # Extreme shock conditions: acute drought + extreme heat OR severe waterlogging + extreme cold
        if (t > 38.0 and m < 20.0) or (m > 88.0 and (h > 92.0 or t < 12.0)):
            labels.append(3)  # Extreme Risk
        # High pathogen risk: sustained high humidity (>80%) with moderate/warm temperatures (18-30°C)
        elif (h >= 78.0 and 16.0 <= t <= 32.0) or (m > 82.0 and h > 75.0):
            labels.append(2)  # High Risk
        # Moderate risk: suboptimal moisture or mild heat/drought stress
        elif (h >= 68.0 and 15.0 <= t <= 35.0) or (m < 30.0) or (m > 75.0) or (t > 34.0):
            labels.append(1)  # Moderate Risk
        else:
            labels.append(0)  # Low Risk (Optimal agronomic envelope)

    df = pd.DataFrame({
        "temperature": temps,
        "humidity": humids,
        "soil_moisture": moists,
        "vpd": vpds,
        "risk_category": labels
    })

    return df


def train_environmental_model() -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Trains and saves the Random Forest environmental disease risk classifier.
    """
    df = generate_agronomic_dataset(n_samples=2500, seed=42)
    X = df[["temperature", "humidity", "soil_moisture", "vpd"]]
    y = df["risk_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(rf, MODEL_SAVE_PATH)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "feature_importances": {
            feat: float(imp) for feat, imp in zip(X.columns, rf.feature_importances_)
        }
    }

    print(f"Environmental Random Forest trained successfully.")
    print(f"Accuracy: {acc * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Feature Importances: {metrics['feature_importances']}")

    return rf, metrics


def load_environmental_model() -> RandomForestClassifier:
    """Loads the trained Random Forest model from disk, training if missing."""
    if os.path.exists(MODEL_SAVE_PATH):
        return joblib.load(MODEL_SAVE_PATH)
    else:
        rf, _ = train_environmental_model()
        return rf


def analyze_environmental_conditions(
    temperature: float,
    humidity: float,
    soil_moisture: float,
    data_source: str = "simulated"
) -> Dict[str, Any]:
    """
    Analyzes specific environmental parameters and returns risk status and actionable indicators.
    """
    rf = load_environmental_model()
    vpd = calculate_vpd(temperature, humidity)

    X_sample = pd.DataFrame([{
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "vpd": vpd
    }])

    pred_class = int(rf.predict(X_sample)[0])
    probs = rf.predict_proba(X_sample)[0]

    risk_labels = {
        0: {"level": "Low", "badge": "Optimal", "color": "green", "score": 15.0},
        1: {"level": "Moderate", "badge": "Caution", "color": "yellow", "score": 45.0},
        2: {"level": "High", "badge": "Disease Alert", "color": "orange", "score": 80.0},
        3: {"level": "Extreme", "badge": "Severe Hazard", "color": "red", "score": 95.0},
    }

    info = risk_labels[pred_class]
    # Smooth score with predicted class probability
    calibrated_score = (info["score"] * 0.7) + (probs[pred_class] * 30.0)

    # Detailed parameter evaluations
    temp_status = "Optimal" if 18.0 <= temperature <= 28.0 else ("Heat Stress" if temperature > 28.0 else "Cold Stress")
    humidity_status = "Pathogen Favorable" if humidity > 75.0 else ("Optimal" if 45.0 <= humidity <= 75.0 else "Dry / Deficit")
    moisture_status = "Waterlogged" if soil_moisture > 75.0 else ("Optimal" if 40.0 <= soil_moisture <= 75.0 else "Drought Stress")

    return {
        "data_source": data_source,
        "risk_category_id": pred_class,
        "risk_level": info["level"],
        "risk_badge": info["badge"],
        "risk_color": info["color"],
        "environmental_risk_score": round(float(calibrated_score), 1),
        "vpd_kpa": round(vpd, 2),
        "class_probabilities": [round(float(p), 3) for p in probs],
        "diagnostics": {
            "temperature_status": temp_status,
            "humidity_status": humidity_status,
            "soil_moisture_status": moisture_status,
        }
    }


def compute_unified_crop_health(
    vision_prediction: Dict[str, Any],
    environmental_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fuses deep learning visual disease predictions with environmental risk analytics
    into a comprehensive Crop Health Assessment.

    Returns:
      - crop_health_index (0 - 100): 100 = Peak vitality, 0 = Total destruction.
      - overall_status: "Healthy", "Vulnerable", "Diseased - Controlled", "Critical Outbreak"
      - actionable_recommendations: List of agronomic treatment and management steps.
    """
    pred_class = vision_prediction.get("predicted_class", "Unknown")
    confidence = vision_prediction.get("confidence", 0.8)
    is_healthy = "healthy" in pred_class.lower()

    env_risk_score = environmental_analysis.get("environmental_risk_score", 30.0)
    env_level = environmental_analysis.get("risk_level", "Moderate")

    # Base Health Index from vision
    if is_healthy:
        # Starts at 95, penalize if environment poses imminent outbreak threat
        base_chi = 95.0 - (env_risk_score * 0.35)
    else:
        # Diseased: Base health reflects disease severity, worsened by favorable pathogen climate
        severity_penalty = 50.0 * confidence
        climatic_acceleration = env_risk_score * 0.30
        base_chi = max(10.0, 90.0 - severity_penalty - climatic_acceleration)

    chi = max(5.0, min(100.0, round(base_chi, 1)))

    # Overall Status Classification
    if chi >= 80.0:
        status = "Healthy & Thriving"
        status_color = "green"
    elif chi >= 60.0:
        status = "Vulnerable / Suboptimal Conditions"
        status_color = "yellow"
    elif chi >= 40.0:
        status = "Moderate Disease Infection"
        status_color = "orange"
    else:
        status = "Critical Pathogen Outbreak"
        status_color = "red"

    # Prescriptive Agronomic Recommendations
    recommendations = []

    if is_healthy:
        if env_level in ["High", "Extreme"]:
            recommendations.append("Preventive Bio-Fungicide: Apply Bacillus subtilis or neem oil spray to shield healthy foliage from high ambient spore pressure.")
            recommendations.append("Canopy Aeration: Prune dense lower canopy to improve air circulation and reduce leaf wetness duration.")
        else:
            recommendations.append("Routine Monitoring: Maintain standard irrigation and scouting intervals; canopy displays robust vigor.")
    else:
        # Class-specific remedies
        if "scab" in pred_class.lower():
            recommendations.append("Fungicide Application: Apply Captan, Mancozeb, or copper-based fungicide at recommended dilution.")
            recommendations.append("Sanitation: Remove and dispose of fallen infected leaves to break overwintering ascospores.")
        elif "black_rot" in pred_class.lower():
            recommendations.append("Canopy Sanitation: Prune dead twigs, mummified fruit, and cankers at least 15 cm below infected tissue.")
            recommendations.append("Fungicide Treatment: Apply myclobutanil or thiophanate-methyl during active vegetative flushes.")
        elif "rust" in pred_class.lower():
            recommendations.append("Host Separation: Eliminate nearby eastern red cedar / juniper alternate hosts within 200m if feasible.")
            recommendations.append("Targeted Spray: Apply sterol-inhibiting fungicide (e.g. Immunox) during early spring gall maturation.")
        elif "early_blight" in pred_class.lower():
            recommendations.append("Foliar Fungicide: Apply Chlorothalonil or Azoxystrobin targeting concentric ring lesions.")
            recommendations.append("Mulching & Drip Irrigation: Avoid overhead sprinkler irrigation to prevent soil-splashing fungal conidia onto lower leaves.")
        else:
            recommendations.append("Broad-Spectrum Fungicide: Apply bio-organic copper hydroxide spray.")

    # Environmental Adjustments
    diag = environmental_analysis.get("diagnostics", {})
    if diag.get("humidity_status") == "Pathogen Favorable":
        recommendations.append("Humidity Management: Elevate greenhouse ventilation or spacing to drop relative humidity below 70%.")
    if diag.get("soil_moisture_status") == "Waterlogged":
        recommendations.append("Drainage Action: Check soil tiling and suspend irrigation cycle until field moisture stabilizes below 70%.")
    elif diag.get("soil_moisture_status") == "Drought Stress":
        recommendations.append("Irrigation Compensation: Deliver deep root zone hydration in early morning to relieve thermal and moisture stress.")

    return {
        "crop_health_index": chi,
        "overall_status": status,
        "status_color": status_color,
        "is_healthy_detected": is_healthy,
        "environmental_impact": f"{env_level} microclimate pressure ({env_risk_score}/100)",
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    print("Training environmental analysis Random Forest model...")
    rf_model, metrics = train_environmental_model()
    print("Testing sample prediction under humid conditions (26°C, 88% RH, 78% moisture)...")
    sample_env = analyze_environmental_conditions(26.0, 88.0, 78.0, data_source="simulated")
    print("Environmental Analysis:", sample_env)
