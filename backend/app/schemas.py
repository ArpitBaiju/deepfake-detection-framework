from pydantic import BaseModel, Field


class ProbabilityBreakdown(BaseModel):
    fake: float = Field(..., description="Probability of FAKE class (class 0)")
    real: float = Field(..., description="Probability of REAL class (class 1)")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted label: REAL or FAKE")
    class_id: int = Field(
        ..., description="Predicted class index: 0 for FAKE, 1 for REAL"
    )
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    model: str = Field(..., description="Selected model identifier")
    processing_time_ms: float = Field(
        ..., description="Inference latency in milliseconds"
    )
    probabilities: ProbabilityBreakdown = Field(
        ..., description="Class probabilities breakdown"
    )
    is_mock: bool = Field(
        False, description="Whether this prediction was simulated due to mock mode"
    )
    device_used: str = Field("cpu", description="Execution device (cuda or cpu)")


class ModelInfo(BaseModel):
    id: str
    name: str
    role: str
    description: str
    architecture_details: str
    target_size: list[int]
    weights_available: bool
    resolved_weight_path: str | None = None
    expected_filename: str


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]
    weights_directory: str
    mock_mode: bool


class HealthResponse(BaseModel):
    status: str
    device: str
    torch_version: str
    timm_version: str | None = None
    mock_mode: bool
    weights_directory: str
    models_ready: dict[str, bool]
