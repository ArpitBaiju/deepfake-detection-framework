import logging
import time

import torch
import torch.nn.functional as F

from app.models.registry import registry
from app.schemas import PredictionResponse, ProbabilityBreakdown
from app.services.preprocessing import load_and_preprocess_image

logger = logging.getLogger(__name__)


def run_inference(file_bytes: bytes, model_id: str) -> PredictionResponse:
    """
    Runs end-to-end inference on the input image using the specified model.

    Args:
        file_bytes: Raw image file bytes.
        model_id: Key identifying the model ('basic_cnn', 'resnet18', etc.).

    Returns:
        PredictionResponse with predicted class, confidence, latency, and probabilities.
    """
    start_time = time.perf_counter()

    # Preprocess image into [1, 3, 256, 256] tensor
    tensor, _ = load_and_preprocess_image(file_bytes)

    # Fetch loaded model instance from registry
    model, is_mock = registry.get_model(model_id)

    # Move tensor to target device (CUDA or CPU)
    tensor = tensor.to(registry.device)

    # Perform forward pass in eval mode with no_grad
    model.eval()
    with torch.no_grad():
        output = model(tensor)

        # Handle 1-output logit (Binary cross-entropy: 0=Fake, 1=Real)
        if output.numel() == 1 or output.shape[-1] == 1:
            real_prob = float(torch.sigmoid(output).squeeze().item())
            fake_prob = 1.0 - real_prob
        elif output.shape[-1] == 2:
            probabilities = F.softmax(output, dim=1)[0]
            fake_prob = float(probabilities[0].item())
            real_prob = float(probabilities[1].item())
        else:
            raise ValueError(f"Unexpected model output shape: {output.shape}")

    # Clamp probabilities to [0.0, 1.0]
    real_prob = max(0.0, min(1.0, real_prob))
    fake_prob = max(0.0, min(1.0, fake_prob))

    # Determine predicted class (0 = FAKE, 1 = REAL)
    if real_prob >= 0.5:
        class_id = 1
        prediction_label = "REAL"
        confidence = real_prob
    else:
        class_id = 0
        prediction_label = "FAKE"
        confidence = fake_prob

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        prediction=prediction_label,
        class_id=class_id,
        confidence=round(confidence, 4),
        model=model_id,
        processing_time_ms=round(latency_ms, 2),
        probabilities=ProbabilityBreakdown(
            fake=round(fake_prob, 4),
            real=round(real_prob, 4),
        ),
        is_mock=is_mock,
        device_used=str(registry.device),
    )
