"""
DAF-Net Modular Blocks for V1, V2, V3, and V4.

V1:
- Local 3x3 Conv vs Context 5x5 Conv
- Deviation D = |F_local - F_context|
- Artifact Gate G = Sigmoid(1x1 Conv(Norm(ReLU(D)))) -> shape (B, C, H, W)
- Residual Amplification F_out = F + alpha * (F * G) (alpha is learnable scalar init to 1.0)

V2:
- Refined Artifact Deviation & Channel/Spatial Separation
- Depthwise-separable convolutions to isolate spatial inconsistencies cleanly
- Dual gating: Spatial deviation gate multiplied with channel-wise squeeze-and-excitation gate

V3:
- Frequency-aware gating branch
- High-frequency residual (Laplacian / High-pass kernel) fused with spatial deviation

V4:
- Adaptive instance-conditioned amplification alpha(x) instead of fixed scalar
- Generated via Global Average Pooling + MLP: alpha = Sigmoid(MLP(GAP(F))) * 2.0 (bounded)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class DownsampleBlock(nn.Module):
    """
    Downsamples spatial resolution by 2x while projecting channels.
    Combines Conv2d (stride 2) + BatchNorm + LeakyReLU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ==========================================
# Artifact Block V1
# ==========================================
class ArtifactBlockV1(nn.Module):
    """
    Basic Artifact Block V1:
    - 3x3 local branch vs 5x5 context branch
    - D = |F_local - F_context|
    - G = Sigmoid(1x1 Conv(BatchNorm(ReLU(D)))) -> (B, C, H, W)
    - F_out = F + alpha * (F * G), alpha is learnable scalar initialized to 1.0
    """
    def __init__(self, channels: int, init_alpha: float = 1.0):
        super().__init__()
        self.channels = channels
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor, return_artifacts: bool = False) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        f_local = self.local_branch(x)
        f_context = self.context_branch(x)
        
        deviation = torch.abs(f_local - f_context)
        gate = self.gate_conv(deviation)
        out = x + self.alpha * (x * gate)
        
        if return_artifacts:
            return out, {
                "deviation": deviation,
                "gate": gate,
                "alpha": self.alpha.item()
            }
        return out


# ==========================================
# Artifact Block V2: Refined Gating & Inconsistency Isolation
# ==========================================
class ArtifactBlockV2(nn.Module):
    """
    Artifact Block V2:
    - Depthwise-separable convolutions for local & context branches to disentangle spatial inconsistency
    - Dual gating: Spatial deviation gate * Channel squeeze-and-excitation gate
    - Residual Amplification with learnable alpha
    """
    def __init__(self, channels: int, init_alpha: float = 1.0):
        super().__init__()
        self.channels = channels
        
        # Depthwise-separable local (3x3)
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise-separable context (5x5)
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial gating from deviation
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # Channel gating from deviation
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, max(channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor, return_artifacts: bool = False) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        f_local = self.local_branch(x)
        f_context = self.context_branch(x)
        deviation = torch.abs(f_local - f_context)
        
        # Dual Gate: (B, 1, H, W) * (B, C, 1, 1) -> (B, C, H, W)
        g_spatial = self.spatial_gate(deviation)
        g_channel = self.channel_gate(deviation)
        gate = g_spatial * g_channel
        
        out = x + self.alpha * (x * gate)
        
        if return_artifacts:
            return out, {
                "deviation": deviation,
                "gate": gate,
                "alpha": self.alpha.item()
            }
        return out


# ==========================================
# Artifact Block V3: Frequency-Aware Gating
# ==========================================
class ArtifactBlockV3(nn.Module):
    """
    Artifact Block V3:
    - Adds high-frequency domain representation via fixed Laplacian / high-pass filtering
    - Combines spatial deviation D_spatial with frequency residual D_freq
    - Learnable fusion of spatial and frequency inconsistency
    """
    def __init__(self, channels: int, init_alpha: float = 1.0):
        super().__init__()
        self.channels = channels
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Fixed high-pass Laplacian filter
        laplacian_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("laplacian_kernel", laplacian_kernel)
        
        # Fusion of spatial deviation + frequency deviation
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor, return_artifacts: bool = False) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        f_local = self.local_branch(x)
        f_context = self.context_branch(x)
        d_spatial = torch.abs(f_local - f_context)
        
        # High-frequency extraction
        d_freq = torch.abs(F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.channels))
        
        # Fuse spatial and frequency deviations
        fused_dev = self.freq_fusion(torch.cat([d_spatial, d_freq], dim=1))
        gate = self.gate_conv(fused_dev)
        
        out = x + self.alpha * (x * gate)
        
        if return_artifacts:
            return out, {
                "deviation": fused_dev,
                "gate": gate,
                "alpha": self.alpha.item()
            }
        return out


# ==========================================
# Artifact Block V4: Adaptive Amplification alpha(x)
# ==========================================
class ArtifactBlockV4(nn.Module):
    """
    Artifact Block V4:
    - Dynamic adaptive amplification alpha(x) per input instance instead of a static scalar.
    - Uses Global Average Pooling + MLP to generate an instance-specific amplification weight
      bounded in [0.0, 2.0].
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.context_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # Adaptive alpha generator
        self.adaptive_alpha = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, max(channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, return_artifacts: bool = False) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        f_local = self.local_branch(x)
        f_context = self.context_branch(x)
        deviation = torch.abs(f_local - f_context)
        gate = self.gate_conv(deviation)
        
        # Adaptive alpha: bounded in [0.0, 2.0]
        alpha = self.adaptive_alpha(x) * 2.0  # Shape: (B, 1, 1, 1)
        
        out = x + alpha * (x * gate)
        
        if return_artifacts:
            return out, {
                "deviation": deviation,
                "gate": gate,
                "alpha": alpha.squeeze().mean().item()
            }
        return out
