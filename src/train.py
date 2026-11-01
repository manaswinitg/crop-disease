"""
Training engine for Crop Health Deep Learning Models.

Trains:
1. Custom CropHealthCNN
2. MobileNetV2 Transfer Learning Architecture

Features:
- Reproducible seeding
- AdamW optimizer with cosine learning rate scheduling
- CrossEntropy loss with label smoothing
- Real-time training and validation metric tracking
- Best checkpoint serialization
- Training history persistence for evaluation curves
"""

import os
import sys
import time
import json
import copy
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.preprocessing import create_dataloaders, set_seed, SEED
from src.models import CropHealthCNN, get_mobilenet_v2_model


def train_single_model(
    model: nn.Module,
    model_name: str,
    dataloaders: Dict[str, Any],
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Executes training and validation loops with model checkpointing.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n==========================================")
    print(f"Starting Training: {model_name}")
    print(f"Device: {device} | Epochs: {epochs} | Initial LR: {lr}")
    print(f"==========================================")

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_samples += inputs.size(0)

        val_loss = val_loss / val_samples
        val_acc = val_corrects / val_samples

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["epoch_times"].append(round(epoch_duration, 2))

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Checkpoint if validation improved
        if val_loss < best_val_loss or (val_loss == best_val_loss and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_model_wts, save_path)
                print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time/60:.2f} mins. Best Val Acc: {best_val_acc * 100:.2f}%, Best Val Loss: {best_val_loss:.4f}")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model, history


def run_training_pipeline(epochs: int = 10, batch_size: int = 16):
    """
    Trains both Custom CNN and MobileNetV2 models and logs histories.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initiating training pipeline on device: {device}")

    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, class_names = create_dataloaders(str(dataset_dir), batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Dataset classes ({num_classes}): {class_names}")

    # 1. Train Custom CNN
    cnn_model = CropHealthCNN(num_classes=num_classes).to(device)
    cnn_save_path = str(models_dir / "custom_cnn_best.pth")
    trained_cnn, cnn_history = train_single_model(
        model=cnn_model,
        model_name="Custom CropHealthCNN",
        dataloaders=dataloaders,
        device=device,
        epochs=epochs,
        lr=8e-4,
        save_path=cnn_save_path
    )

    # 2. Train MobileNetV2 Transfer Learning
    mobilenet_model = get_mobilenet_v2_model(num_classes=num_classes, pretrained=True).to(device)
    mobilenet_save_path = str(models_dir / "mobilenet_best.pth")
    trained_mobilenet, mobilenet_history = train_single_model(
        model=mobilenet_model,
        model_name="MobileNetV2 (Transfer Learning)",
        dataloaders=dataloaders,
        device=device,
        epochs=epochs,
        lr=5e-4,
        save_path=mobilenet_save_path
    )

    # Save training histories
    training_summary = {
        "classes": class_names,
        "epochs": epochs,
        "custom_cnn": cnn_history,
        "mobilenet_v2": mobilenet_history,
    }

    with open(outputs_dir / "training_history.json", "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\nAll models trained and saved successfully.")
    print(f"Custom CNN saved at: {cnn_save_path}")
    print(f"MobileNetV2 saved at: {mobilenet_save_path}")
    print(f"Training history saved at: {outputs_dir / 'training_history.json'}")


if __name__ == "__main__":
    run_training_pipeline(epochs=10, batch_size=16)
