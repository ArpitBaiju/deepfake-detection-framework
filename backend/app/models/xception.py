import logging

from torch import nn

logger = logging.getLogger(__name__)


def build_xception(num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    """
    Build Xception architecture for deepfake binary classification.
    Matches trained weights: timm legacy_xception with 1-logit output.
    """
    import timm

    model = timm.create_model(
        "legacy_xception", pretrained=pretrained, num_classes=num_classes
    )
    return model
