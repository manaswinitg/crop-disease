"""
Inference Pipeline and Explainable AI (Grad-CAM) Module.

Features:
- Dual-model inference (MobileNetV2 or Custom CNN)
- Gradient-weighted Class Activation Mapping (Grad-CAM)
- Tri-panel visualization: Original Leaf, Activation Heatmap, and Alpha Overlay
- End-to-end diagnosis with confidence breakdown
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from src.models import CropHealthCNN, get_mobilenet_v2_model
from src.preprocessing import preprocess_single_image, NORM_MEAN, NORM_STD

MODELS_DIR = PROJECT_ROOT / "models"


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Hooks into the final convolutional feature extractor to produce class-discriminative saliency maps.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class_idx: Optional[int] = None) -> np.ndarray:
        """
        Computes the 2D normalized Grad-CAM heatmap for a given input tensor.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        # Target score for backprop
        score = logits[0, target_class_idx]
        score.backward()

        # Global average pooling of gradients
        # Gradients shape: (1, C, H, W)
        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # Shape: (1, 1, H, W)
        cam = F.relu(cam)  # Apply ReLU to focus on features with positive influence

        cam = cam.squeeze().cpu().numpy()

        # Normalize between 0 and 1
        if np.max(cam) - np.min(cam) > 1e-8:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        else:
            cam = np.zeros_like(cam)

        return cam


class CropHealthPredictor:
    """
    Manages model loading, inference execution, and Grad-CAM visualization generation.
    """
    def __init__(self, default_model: str = "mobilenet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_classes()
        self.models = {}
        self.gradcams = {}
        self.default_model_type = default_model

        # Pre-load default model
        self.get_model(default_model)

    def _load_classes(self) -> list:
        class_file = MODELS_DIR / "class_names.json"
        if class_file.exists():
            with open(class_file, "r") as f:
                return json.load(f)
        return [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Potato___Early_blight",
            "Potato___healthy",
        ]

    def get_model(self, model_type: str = "mobilenet") -> Tuple[nn.Module, GradCAM]:
        """Loads and caches the requested model architecture and its Grad-CAM hook."""
        model_type = model_type.lower()
        if model_type in self.models:
            return self.models[model_type], self.gradcams[model_type]

        num_classes = len(self.classes)

        if "mobilenet" in model_type:
            model = get_mobilenet_v2_model(num_classes=num_classes, pretrained=False).to(self.device)
            ckpt_path = MODELS_DIR / "mobilenet_best.pth"
            target_layer = model.features[-1]
        else:
            model = CropHealthCNN(num_classes=num_classes).to(self.device)
            ckpt_path = MODELS_DIR / "custom_cnn_best.pth"
            target_layer = model.features[-3]  # Final ConvBlock

        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"Loaded weights for {model_type} from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Running initialized model.")

        model.eval()
        gradcam = GradCAM(model, target_layer)

        self.models[model_type] = model
        self.gradcams[model_type] = gradcam
        return model, gradcam

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        model_type: str = "mobilenet",
        generate_cam: bool = True
    ) -> Dict[str, Any]:
        """
        Executes disease diagnosis and Grad-CAM visual explanation.
        """
        # Load raw PIL image for overlay creation
        if isinstance(image_input, str):
            raw_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            raw_img = image_input.convert("RGB")
        else:
            raw_img = Image.fromarray(image_input).convert("RGB")

        # Preprocess tensor
        input_tensor = preprocess_single_image(raw_img).to(self.device)
        input_tensor.requires_grad = True

        model, gradcam = self.get_model(model_type)

        # Inference
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        is_healthy = "healthy" in pred_class.lower()

        # Class-wise probabilities
        all_probs = {
            self.classes[i]: round(float(probs[i]), 4)
            for i in range(len(self.classes))
        }

        # Human-friendly class name
        display_name = pred_class.replace("___", " - ").replace("_", " ")

        result = {
            "predicted_class": pred_class,
            "display_name": display_name,
            "confidence": round(confidence, 4),
            "is_healthy": is_healthy,
            "model_type": model_type,
            "all_probabilities": all_probs,
        }

        if generate_cam:
            cam_map = gradcam.generate_heatmap(input_tensor, target_class_idx=pred_idx)

            # Resize heatmap to raw image dimensions
            orig_w, orig_h = raw_img.size
            cam_resized = cv2.resize(cam_map, (orig_w, orig_h))

            # Color map (JET)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Overlay on original image
            raw_np = np.array(raw_img)
            overlay = np.uint8(0.55 * raw_np + 0.45 * heatmap_color)

            result["gradcam"] = {
                "heatmap_array": cam_resized,
                "heatmap_image": Image.fromarray(heatmap_color),
                "overlay_image": Image.fromarray(overlay),
                "original_image": raw_img,
            }

        return result


_PREDICTOR_INSTANCE: Optional[CropHealthPredictor] = None


def get_predictor() -> CropHealthPredictor:
    """Singleton getter for the CropHealthPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = CropHealthPredictor()
    return _PREDICTOR_INSTANCE


if __name__ == "__main__":
    predictor = get_predictor()
    # Test with a test image
    test_img_path = PROJECT_ROOT / "dataset" / "test" / "Apple___Black_rot"
    if test_img_path.exists():
        sample_file = list(test_img_path.glob("*.jpg"))[0]
        print(f"Testing Grad-CAM and inference on sample: {sample_file}")
        res = predictor.predict(str(sample_file), model_type="mobilenet", generate_cam=True)
        print(f"Prediction: {res['display_name']} ({res['confidence']*100:.2f}%)")
        print(f"Is Healthy: {res['is_healthy']}")
        print(f"Grad-CAM overlay generated: {res['gradcam']['overlay_image'].size}")
        # Save sample gradcam to outputs
        out_cam = PROJECT_ROOT / "outputs" / "sample_gradcam.png"
        res['gradcam']['overlay_image'].save(out_cam)
        print(f"Sample Grad-CAM overlay saved to {out_cam}")
