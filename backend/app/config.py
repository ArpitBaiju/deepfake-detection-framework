import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", BASE_DIR / "weights"))

# Ensure weights directory exists
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Default model weights mappings
MODEL_WEIGHT_FILENAMES: dict[str, list[str]] = {
    "basic_cnn": ["basic_cnn.pth", "basic_cnn_weights.pth"],
    "resnet18": ["resnet18.pth", "resnet18_weights.pth"],
    "efficientnet_b0": ["efficientnet_b0.pth", "efficientnet.pth"],
    "xception": ["xception_current.pth", "xception.pth", "xception_weights.pth"],
}

MODEL_METADATA = {
    "basic_cnn": {
        "id": "basic_cnn",
        "name": "Basic CNN",
        "role": "Baseline",
        "description": "Baseline convolutional neural network for deepfake feature extraction",
        "architecture_details": "4-block Conv2d with BatchNorm, LeakyReLU, MaxPool2d and Linear classifier",
        "target_size": [256, 256],
    },
    "resnet18": {
        "id": "resnet18",
        "name": "ResNet18",
        "role": "Residual CNN",
        "description": "Uses residual skip connections to improve deep-network training and gradient flow",
        "architecture_details": "18-layer residual architecture with identity shortcut mappings",
        "target_size": [256, 256],
    },
    "efficientnet_b0": {
        "id": "efficientnet_b0",
        "name": "EfficientNet-B0",
        "role": "Efficient architecture",
        "description": "Balances accuracy and computational efficiency via compound coefficient scaling",
        "architecture_details": "MBConv blocks with squeeze-and-excitation optimization",
        "target_size": [256, 256],
    },
    "xception": {
        "id": "xception",
        "name": "Xception",
        "role": "Advanced CNN",
        "description": "Uses depthwise separable convolutions for rich spatial cross-channel representations",
        "architecture_details": "Entry, Middle (8 repeats), and Exit flow with depthwise separable layers",
        "target_size": [256, 256],
    },
}

# Normalization parameters (ImageNet default)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (256, 256)

# Class mappings: 0 = Fake, 1 = Real
CLASS_NAMES = {0: "FAKE", 1: "REAL"}

# Server settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Mock mode for frontend/UI development only when weights are absent
# Defaults strictly to False
MOCK_MODE = os.getenv("MOCK_MODE", "false").strip().lower() in ("true", "1", "yes")
