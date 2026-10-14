# Model Weights Directory

Place your trained PyTorch model weight files (`.pth`) in this directory.

## Expected Files

| Model Name | Expected Filename | Architecture | Class Mapping |
| :--- | :--- | :--- | :--- |
| **Basic CNN** | `basic_cnn.pth` | Custom 4-block ConvNet | `0` = Fake, `1` = Real |
| **ResNet18** | `resnet18.pth` | torchvision.models.resnet18 (num_classes=2) | `0` = Fake, `1` = Real |
| **EfficientNet-B0** | `efficientnet_b0.pth` | torchvision.models.efficientnet_b0 (num_classes=2) | `0` = Fake, `1` = Real |
| **Xception** | `xception_current.pth` *(or `xception.pth`)* | timm or custom Xception (num_classes=2) | `0` = Fake, `1` = Real |

## Input Specifications

All models expect input images with:
- **Spatial Resolution**: `256 × 256`
- **Channels**: `3` (RGB)
- **Normalization**:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`
- **Binary Output**:
  - Class `0`: Fake
  - Class `1`: Real

## Weight State Loading

The backend automatically checks for the existence of these files at startup and when handling requests. If a `.pth` file is missing, the API informs the caller that the weights have not been mounted yet.
