from torch import nn
from torchvision.models import efficientnet_b0


def build_efficientnet_b0(num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    """
    Build EfficientNet-B0 adapted for deepfake binary classification.
    Matches trained weights: torchvision.models.efficientnet_b0 with 1-logit output.
    """
    model = efficientnet_b0(weights=None, num_classes=num_classes)
    return model
