"""
Artifact Map and Feature Visualization utilities for DAF-Net.
Extracts:
1. Block-wise deviation maps D
2. Block-wise artifact gates G
3. Side-by-side comparison between Real and Fake images
"""
import torch
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from typing import Dict, Any, Optional


def visualize_artifact_maps(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    raw_image_np: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Passes a single image (1, 3, H, W) through DAF-Net, extracts intermediate artifact maps,
    and plots the input alongside Deviation and Gate maps for Stage 1, 2, and 3.
    """
    model.eval()
    with torch.no_grad():
        logit, art_maps = model(image_tensor, return_artifact_maps=True)
        prob = torch.sigmoid(logit).item()
        
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 1. Original image
    if raw_image_np is not None:
        axes[0, 0].imshow(raw_image_np)
    else:
        # Denormalize tensor for visualization
        img_vis = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
        axes[0, 0].imshow(img_vis)
    axes[0, 0].set_title(f"Input (Pred Fake Prob: {prob:.3f})")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    
    blocks = ["block1", "block2", "block3"]
    for i, b_name in enumerate(blocks, start=1):
        b_info = art_maps[b_name]
        # Mean across channels for spatial visualization
        dev = b_info["deviation"].squeeze().mean(dim=0).cpu().numpy()
        gate = b_info["gate"].squeeze().mean(dim=0).cpu().numpy()
        alpha = b_info["alpha"]
        
        # Plot deviation
        im_dev = axes[0, i].imshow(dev, cmap="magma")
        axes[0, i].set_title(f"{b_name.capitalize()} Dev |3x3-5x5|")
        axes[0, i].axis("off")
        plt.colorbar(im_dev, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Plot gate
        im_gate = axes[1, i].imshow(gate, cmap="viridis", vmin=0, vmax=1)
        axes[1, i].set_title(f"{b_name.capitalize()} Gate (α={alpha:.2f})")
        axes[1, i].axis("off")
        plt.colorbar(im_gate, ax=axes[1, i], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
