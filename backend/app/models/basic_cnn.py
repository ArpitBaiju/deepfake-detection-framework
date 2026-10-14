import torch
from torch import nn


class BasicCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for Deepfake Binary Classification.
    Matches the trained weights structure: 4-block ConvNet with 1 output logit.
    Class mapping: Fake = 0, Real = 1.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 256x256 -> 128x128
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4: 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_basic_cnn(num_classes: int = 1) -> BasicCNN:
    return BasicCNN(num_classes=num_classes)
