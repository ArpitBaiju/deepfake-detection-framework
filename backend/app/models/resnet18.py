from torch import nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    """
    Build ResNet18 adapted for deepfake binary classification.
    Matches trained weights: torchvision.models.resnet18 with 1-logit output.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    return model
