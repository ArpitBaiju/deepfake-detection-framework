import logging
from pathlib import Path

import torch
from torch import nn

from app.config import (
    MOCK_MODE,
    MODEL_METADATA,
    MODEL_WEIGHT_FILENAMES,
    WEIGHTS_DIR,
)
from app.models.basic_cnn import build_basic_cnn
from app.models.efficientnet import build_efficientnet_b0
from app.models.resnet18 import build_resnet18
from app.models.xception import build_xception

logger = logging.getLogger(__name__)

# Device selection (CUDA if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelNotFoundError(Exception):
    """Raised when an unknown model ID is requested."""


class WeightsNotFoundError(Exception):
    """Raised when the required weight file (.pth) is not present."""


class ModelRegistry:
    """
    Manages model registration, resolution of trained .pth weights,
    and lazy-loading of models into evaluation mode.
    """

    def __init__(self, weights_dir: Path = WEIGHTS_DIR):
        self.weights_dir = Path(weights_dir)
        self.device = DEVICE
        self._builders = {
            "basic_cnn": build_basic_cnn,
            "resnet18": build_resnet18,
            "efficientnet_b0": build_efficientnet_b0,
            "xception": build_xception,
        }
        self._loaded_models: dict[str, nn.Module] = {}
        logger.info("Initialized ModelRegistry using device: %s", self.device)

    def resolve_weight_path(self, model_id: str) -> Path | None:
        """Check if any expected weight file exists for this model."""
        if model_id not in MODEL_WEIGHT_FILENAMES:
            return None

        for filename in MODEL_WEIGHT_FILENAMES[model_id]:
            candidate = self.weights_dir / filename
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def is_model_available(self, model_id: str) -> bool:
        """Returns True if weights exist on disk for the model."""
        return self.resolve_weight_path(model_id) is not None

    def get_models_status(self) -> dict[str, dict]:
        """Returns metadata and availability status for all 4 models."""
        result = {}
        for mid, meta in MODEL_METADATA.items():
            resolved = self.resolve_weight_path(mid)
            expected_file = MODEL_WEIGHT_FILENAMES.get(mid, [""])[0]
            result[mid] = {
                **meta,
                "weights_available": resolved is not None,
                "resolved_weight_path": str(resolved) if resolved else None,
                "expected_filename": expected_file,
            }
        return result

    def _clean_state_dict(self, state_dict: dict) -> dict:
        """Strips 'module.' prefixes and un-nests top-level wrapper keys."""
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                cleaned[k[7:]] = v
            else:
                cleaned[k] = v
        return cleaned

    def get_model(self, model_id: str) -> tuple[nn.Module, bool]:
        """
        Retrieves the model instance loaded into memory.
        Returns: (model_instance, is_mock)
        Raises WeightsNotFoundError if weights are missing and MOCK_MODE is disabled.
        """
        model_id = model_id.lower().strip()
        if model_id not in self._builders:
            raise ModelNotFoundError(
                f"Unknown model '{model_id}'. Supported models are: {list(self._builders.keys())}"
            )

        # Return cached instance if already loaded
        if model_id in self._loaded_models:
            return self._loaded_models[model_id], False

        weight_path = self.resolve_weight_path(model_id)

        # If weights do not exist
        if weight_path is None:
            if MOCK_MODE:
                logger.warning(
                    "MOCK_MODE enabled: Weights not found for '%s'. Initializing unweighted architecture for UI testing.",
                    model_id,
                )
                model = self._builders[model_id](num_classes=1)
                model.to(self.device)
                model.eval()
                return model, True
            else:
                expected_files = ", ".join(MODEL_WEIGHT_FILENAMES.get(model_id, []))
                raise WeightsNotFoundError(
                    f"Model weights for '{model_id}' not found. "
                    f"Please place trained weights file ({expected_files}) inside '{self.weights_dir}'."
                )

        # Load weights into architecture
        logger.info("Loading weights for '%s' from %s", model_id, weight_path)
        model = self._builders[model_id](num_classes=1)

        try:
            checkpoint = torch.load(weight_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                cleaned_state = self._clean_state_dict(checkpoint)
                load_res = model.load_state_dict(cleaned_state, strict=False)
                logger.info("Loaded state dict for %s: %s", model_id, load_res)
            elif isinstance(checkpoint, nn.Module):
                model = checkpoint
            else:
                logger.warning(
                    "Unrecognized checkpoint format for %s, using loaded object",
                    model_id,
                )
        except Exception as exc:
            logger.error("Failed to load checkpoint file %s: %s", weight_path, exc)
            raise RuntimeError(f"Failed to load weights from {weight_path}: {exc}")

        model.to(self.device)
        model.eval()
        self._loaded_models[model_id] = model
        return model, False


# Global singleton instance
registry = ModelRegistry()
