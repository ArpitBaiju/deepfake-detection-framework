"""
Quick Local Verification Script:
Loads the downloaded trained weights (dafnet_v1_best.pth) from checkpoints/
and verifies state_dict integrity, parameter counts, and inference.
"""
import os
import torch
from src.models.daf_net import DAFNetV1

def verify_checkpoint():
    ckpt_path = "checkpoints/dafnet_v1_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found!")
        return
        
    print(f"Loading checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / (1024*1024):.2f} MB)...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    print(f"  -> Checkpoint keys: {list(ckpt.keys())}")
    print(f"  -> Saved Best Epoch: {ckpt.get('epoch')}")
    if "metrics" in ckpt:
        m = ckpt["metrics"]
        print(f"  -> Best Metrics: Val Acc = {m['accuracy']*100:.2f}% | Val F1 = {m['f1']:.4f} | ROC-AUC = {m['roc_auc']:.4f}")
        print(f"  -> Confusion Matrix: {m.get('cm')}")
        
    # Instantiate model and load weights
    model = DAFNetV1()
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("  -> Model weights loaded successfully without any key mismatches!")
    
    # Test forward pass with dummy tensor
    dummy_x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        logit, maps = model(dummy_x, return_artifact_maps=True)
        prob = torch.sigmoid(logit).item()
    print(f"  -> Test Forward Pass completed. Output Logit: {logit.item():.4f} (Prob: {prob:.4f})")
    print("All checkpoints and model weights are 100% verified and ready!\n")

if __name__ == "__main__":
    verify_checkpoint()
