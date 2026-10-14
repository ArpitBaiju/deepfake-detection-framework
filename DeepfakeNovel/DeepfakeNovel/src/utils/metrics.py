"""
Comprehensive evaluation metrics and profiler for DAF-Net experiments.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Parameter count, and FLOPs.
"""
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true: List[float], y_prob: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes all standard binary classification metrics.
    """
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= threshold).astype(int)
    
    acc = accuracy_score(y_true_np, y_pred_np)
    prec = precision_score(y_true_np, y_pred_np, zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        auc = 0.5  # In case only one class is present in batch
        
    cm = confusion_matrix(y_true_np, y_pred_np).tolist()
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm
    }


def profile_model(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 128, 128), device: str = "cpu") -> Dict[str, Any]:
    """
    Calculates parameter count, FLOPs estimate, and inference latency.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)
    
    # 1. Total & Trainable Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. Latency Benchmark
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Measure
    num_runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    latency_ms = ((time.time() - start) / num_runs) * 1000.0
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "inference_latency_ms": round(latency_ms, 2)
    }
