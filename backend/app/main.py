import logging

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    MOCK_MODE,
    WEIGHTS_DIR,
)
from app.models.registry import (
    ModelNotFoundError,
    WeightsNotFoundError,
    registry,
)
from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    PredictionResponse,
)
from app.services.inference import run_inference
from app.services.preprocessing import ImageValidationError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_app")

app = FastAPI(
    title="Deepfake Detection Framework API",
    description="Inference service for facial deepfake classification using PyTorch deep learning models.",
    version="1.0.0",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all during local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and PyTorch runtime diagnostics."""
    models_status = registry.get_models_status()
    models_ready = {
        mid: info["weights_available"] for mid, info in models_status.items()
    }

    timm_version: str | None = None
    try:
        import timm

        timm_version = timm.__version__
    except ImportError:
        pass

    return HealthResponse(
        status="healthy",
        device=str(registry.device),
        torch_version=torch.__version__,
        timm_version=timm_version,
        mock_mode=MOCK_MODE,
        weights_directory=str(WEIGHTS_DIR),
        models_ready=models_ready,
    )


@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """Retrieve all available models, architectural roles, and weight statuses."""
    status_dict = registry.get_models_status()
    models_list = [
        ModelInfo(
            id=info["id"],
            name=info["name"],
            role=info["role"],
            description=info["description"],
            architecture_details=info["architecture_details"],
            target_size=info["target_size"],
            weights_available=info["weights_available"],
            resolved_weight_path=info["resolved_weight_path"],
            expected_filename=info["expected_filename"],
        )
        for info in status_dict.values()
    ]

    return ModelsListResponse(
        models=models_list,
        weights_directory=str(WEIGHTS_DIR),
        mock_mode=MOCK_MODE,
    )


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_deepfake(
    image: UploadFile = File(..., description="Facial image file (JPG, JPEG, PNG)"),
    model: str = Form(
        ...,
        description="Model identifier: basic_cnn, resnet18, efficientnet_b0, or xception",
    ),
):
    """
    Classifies an uploaded facial image as REAL or FAKE using the requested model.

    Returns:
    - prediction: "REAL" or "FAKE"
    - class_id: 0 (FAKE) or 1 (REAL)
    - confidence: Float confidence (0.0 to 1.0)
    - model: Name of the model
    - processing_time_ms: Milliseconds elapsed
    - probabilities: Probability distribution {fake, real}
    """
    filename = (image.filename or "").lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type for '{image.filename}'. Supported formats: JPG, JPEG, PNG.",
        )

    try:
        file_bytes = await image.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read the uploaded image stream.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    # Enforce maximum image size (e.g. 15MB)
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowed limit of 15MB.",
        )

    try:
        response = run_inference(file_bytes=file_bytes, model_id=model)
        return response

    except WeightsNotFoundError as exc:
        logger.warning("Weights not found error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": str(exc),
                "error_code": "WEIGHTS_NOT_FOUND",
                "model": model,
                "weights_directory": str(WEIGHTS_DIR),
            },
        )
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc!s}",
        )
