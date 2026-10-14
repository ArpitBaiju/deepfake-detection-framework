"""
Configuration file for DAF-Net experiments.
Standardizes training protocol across V1-V5 and comparative baselines.
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DAFNetConfig:
    # Model architecture
    model_version: str = "v1"  # "v1", "v2", "v3", "v4", "vanilla"
    in_channels: int = 3
    stem_channels: int = 32
    stage_channels: Tuple[int, int, int] = (64, 128, 256)
    dropout_rate: float = 0.2
    
    # Dataset & Resolution
    # Development: 128x128. Final scaling: 224x224 or 256x256
    image_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    
    # Paths (Default Kaggle paths with fallback)
    kaggle_data_dir: str = "/kaggle/input/deepfake-and-real-images"
    local_data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    
    # Fixed Training Protocol
    seed: int = 42
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    min_lr: float = 1e-6
    patience: int = 5  # Early stopping patience
    mixed_precision: bool = True
    
    # V1 specifics
    init_alpha: float = 1.0
