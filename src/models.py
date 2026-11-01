"""
Model architectures for Crop Health Monitoring System:
1. CropHealthCNN: Custom deep convolutional neural network with BatchNorm, Dropout, and residual-style conv blocks.
2. CropHealthMobileNetV2: Transfer learning architecture based on MobileNetV2 for edge and real-time deployment.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ConvBlock(nn.Module):
    """Convolutional building block with Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CropHealthCNN(nn.Module):
    """
    Custom Deep Convolutional Neural Network for Crop Disease Classification.
    Architecture:
    - 4 Conv Blocks with progressive feature mapping (3 -> 32 -> 64 -> 128 -> 256)
    - Batch Normalization after every convolution
    - Max Pooling for spatial dimension downsampling
    - Dropout regularizers at each stage to prevent overfitting
    - Global Adaptive Pooling + Fully Connected layers
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.15),

            # Stage 2: 112x112 -> 56x56
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.20),

            # Stage 3: 56x56 -> 28x28
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Stage 4: 28x28 -> 14x14
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.30),
        )

        # Adaptive pooling to fixed spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits


def get_mobilenet_v2_model(num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """
    Constructs a MobileNetV2 transfer learning model with ImageNet pre-trained weights.
    Custom classification head is appended for crop health diagnosis.
    """
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    # Freeze base feature extractor parameters initially or allow fine-tuning
    # We fine-tune the deeper inverted residual blocks and the classifier
    for param in model.features[:-4].parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    return model
