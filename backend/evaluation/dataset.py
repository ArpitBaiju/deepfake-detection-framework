import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from app.services.preprocessing import inference_transform

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EvaluationSample:
    def __init__(self, path: Path, label: int):
        self.path = path
        self.label = label  # 0 for FAKE, 1 for REAL


class DeepfakeEvaluationDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Model Evaluation.
    Parses a directory structured as:
        dataset_dir/
            fake/ -> Class 0
            real/ -> Class 1
    Applies the exact same transformation pipeline as the production inference API.
    """

    def __init__(self, dataset_dir: Path, validate_on_init: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.samples: List[EvaluationSample] = []
        self.corrupted_files: List[Tuple[Path, str]] = []
        self.transform = inference_transform

        self._load_samples(validate=validate_on_init)

    @staticmethod
    def _find_folder(base_dir: Path, names: List[str]) -> Optional[Path]:
        for name in names:
            target = base_dir / name
            if target.is_dir():
                return target
        if base_dir.is_dir():
            name_lowers = [n.lower() for n in names]
            for child in base_dir.iterdir():
                if child.is_dir() and child.name.lower() in name_lowers:
                    return child
        return None

    def _load_samples(self, validate: bool = True):
        self.samples.clear()
        self.corrupted_files.clear()

        search_dir = self.dataset_dir
        # If search_dir doesn't directly contain fake/real, check if it contains test/ or Test/
        fake_folder = self._find_folder(search_dir, ["fake", "fakes"])
        real_folder = self._find_folder(search_dir, ["real", "reals"])

        if not fake_folder and not real_folder:
            test_sub = self._find_folder(search_dir, ["test", "val", "validation"])
            if test_sub:
                logger.info("Auto-detected test split in: %s", test_sub)
                search_dir = test_sub
                fake_folder = self._find_folder(search_dir, ["fake", "fakes"])
                real_folder = self._find_folder(search_dir, ["real", "reals"])

        class_dirs = []
        if fake_folder:
            class_dirs.append((fake_folder, 0))
        if real_folder:
            class_dirs.append((real_folder, 1))

        if not class_dirs:
            logger.warning("No fake/ or real/ subdirectories found in: %s", search_dir)

        for folder, label in class_dirs:
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    if validate:
                        try:
                            with Image.open(file_path) as img:
                                img.verify()
                            self.samples.append(EvaluationSample(file_path, label))
                        except Exception as exc:
                            logger.warning("Corrupted image detected: %s (%s)", file_path, exc)
                            self.corrupted_files.append((file_path, str(exc)))
                    else:
                        self.samples.append(EvaluationSample(file_path, label))

        logger.info(
            "Loaded %d evaluation samples from %s (corrupted: %d)",
            len(self.samples),
            search_dir,
            len(self.corrupted_files),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        image = Image.open(sample.path)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image)
        return tensor, sample.label, str(sample.path)

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistical distribution of the dataset."""
        num_fake = sum(1 for s in self.samples if s.label == 0)
        num_real = sum(1 for s in self.samples if s.label == 1)
        total = len(self.samples)

        return {
            "dataset_path": str(self.dataset_dir),
            "total_samples": total,
            "num_fake": num_fake,
            "num_real": num_real,
            "fake_percentage": round((num_fake / total * 100), 2) if total > 0 else 0.0,
            "real_percentage": round((num_real / total * 100), 2) if total > 0 else 0.0,
            "num_corrupted": len(self.corrupted_files),
            "input_resolution": [256, 256],
            "class_mapping": {"0": "FAKE", "1": "REAL"},
        }
