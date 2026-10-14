import io
import logging

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from app.config import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD

logger = logging.getLogger(__name__)

# Standard preprocessing transform matching training specifications
inference_transform = transforms.Compose(
    [
        transforms.Resize(
            IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]
)


class ImageValidationError(Exception):
    """Raised when an uploaded file is not a valid or readable image."""


def load_and_preprocess_image(
    file_bytes: bytes,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Validates, loads, and preprocesses image bytes for PyTorch inference.

    Args:
        file_bytes: Raw bytes of the uploaded image.

    Returns:
        Tuple of (preprocessed_tensor, original_dimensions)
        where preprocessed_tensor has shape [1, 3, 256, 256].
    """
    if not file_bytes:
        raise ImageValidationError("Uploaded image file is empty.")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Handle EXIF orientation if present
        image = ImageOps.exif_transpose(image)
        original_size = image.size  # (width, height)
    except Exception as exc:
        logger.error("Failed to parse image with Pillow: %s", exc)
        raise ImageValidationError(
            "The file could not be decoded as an image. Please provide a valid JPG or PNG."
        )

    # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply PyTorch transformations
    tensor = inference_transform(image)

    # Add batch dimension: [3, 256, 256] -> [1, 3, 256, 256]
    tensor = tensor.unsqueeze(0)

    return tensor, original_size
