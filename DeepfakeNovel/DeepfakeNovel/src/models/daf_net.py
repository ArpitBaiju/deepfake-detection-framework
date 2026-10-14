"""
Unified DAF-Net Model Factory supporting V1, V2, V3, V4, and Ablations.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.models.blocks import (
    DownsampleBlock,
    ArtifactBlockV1,
    ArtifactBlockV2,
    ArtifactBlockV3,
    ArtifactBlockV4
)


class DAFNet(nn.Module):
    """
    Parametric DAF-Net Architecture supporting versions:
    - 'v1': Standard ArtifactBlockV1 (3x3 vs 5x5, learnable scalar alpha)
    - 'v2': Refined ArtifactBlockV2 (depthwise-separable, dual spatial+channel gate)
    - 'v3': Frequency-aware ArtifactBlockV3 (Laplacian high-pass residual)
    - 'v4': Adaptive amplification ArtifactBlockV4 (instance-conditioned alpha(x))
    """
    def __init__(
        self,
        version: str = "v1",
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int] = (64, 128, 256),
        dropout_rate: float = 0.2,
        init_alpha: float = 1.0
    ):
        super().__init__()
        self.version = version.lower()
        c1, c2, c3 = stage_channels
        
        # Feature Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(stem_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Instantiate Artifact Blocks according to version
        if self.version == "v1":
            self.ab1 = ArtifactBlockV1(c1, init_alpha=init_alpha)
            self.ab2 = ArtifactBlockV1(c2, init_alpha=init_alpha)
            self.ab3 = ArtifactBlockV1(c3, init_alpha=init_alpha)
        elif self.version == "v2":
            self.ab1 = ArtifactBlockV2(c1, init_alpha=init_alpha)
            self.ab2 = ArtifactBlockV2(c2, init_alpha=init_alpha)
            self.ab3 = ArtifactBlockV2(c3, init_alpha=init_alpha)
        elif self.version == "v3":
            self.ab1 = ArtifactBlockV3(c1, init_alpha=init_alpha)
            self.ab2 = ArtifactBlockV3(c2, init_alpha=init_alpha)
            self.ab3 = ArtifactBlockV3(c3, init_alpha=init_alpha)
        elif self.version == "v4":
            self.ab1 = ArtifactBlockV4(c1)
            self.ab2 = ArtifactBlockV4(c2)
            self.ab3 = ArtifactBlockV4(c3)
        else:
            raise ValueError(f"Unknown DAFNet version: {version}. Choose from ['v1', 'v2', 'v3', 'v4'].")
            
        self.ds1 = DownsampleBlock(in_channels=c1, out_channels=c2)
        self.ds2 = DownsampleBlock(in_channels=c2, out_channels=c3)
        
        # Classifier Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(c3, 1)

    def forward(self, x: torch.Tensor, return_artifact_maps: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        feat = self.stem(x)
        
        # Stage 1
        if return_artifact_maps:
            feat, art1 = self.ab1(feat, return_artifacts=True)
        else:
            feat = self.ab1(feat)
        feat = self.ds1(feat)
        
        # Stage 2
        if return_artifact_maps:
            feat, art2 = self.ab2(feat, return_artifacts=True)
        else:
            feat = self.ab2(feat)
        feat = self.ds2(feat)
        
        # Stage 3
        if return_artifact_maps:
            feat, art3 = self.ab3(feat, return_artifacts=True)
        else:
            feat = self.ab3(feat)
            
        pooled = self.gap(feat).flatten(1)
        dropped = self.dropout(pooled)
        logit = self.fc(dropped)
        
        if return_artifact_maps:
            return logit, {
                "block1": art1,
                "block2": art2,
                "block3": art3
            }
            
        return logit


# Backward-compatible alias
DAFNetV1 = lambda **kwargs: DAFNet(version="v1", **kwargs)
DAFNetV2 = lambda **kwargs: DAFNet(version="v2", **kwargs)
DAFNetV3 = lambda **kwargs: DAFNet(version="v3", **kwargs)
DAFNetV4 = lambda **kwargs: DAFNet(version="v4", **kwargs)


class VanillaBaselineCNN(nn.Module):
    """
    Identical CNN backbone without Artifact Blocks, used as the pure baseline for ablation.
    Replaces each ArtifactBlock with a standard Conv3x3-BN-ReLU layer having equivalent parameter footprint.
    """
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 32,
        stage_channels: Tuple[int, int, int] = (64, 128, 256),
        dropout_rate: float = 0.2
    ):
        super().__init__()
        c1, c2, c3 = stage_channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(stem_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.downsample1 = DownsampleBlock(c1, c2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.downsample2 = DownsampleBlock(c2, c3)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(c3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.conv1(feat)
        feat = self.downsample1(feat)
        feat = self.conv2(feat)
        feat = self.downsample2(feat)
        feat = self.conv3(feat)
        pooled = self.gap(feat).flatten(1)
        dropped = self.dropout(pooled)
        return self.classifier(dropped)
