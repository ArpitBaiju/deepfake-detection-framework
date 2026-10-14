"""
Standardized training and validation engine with early stopping, mixed precision, and metric logging.
Follows the exact fixed protocol for DAF-Net.
"""
import os
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from src.utils.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        patience: int = 5,
        checkpoint_dir: str = "./checkpoints",
        model_name: str = "daf_net_v1",
        device: Optional[str] = None,
        use_amp: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.use_amp = use_amp and (self.device == "cuda")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler: Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        self.scaler = GradScaler(enabled=self.use_amp)
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_accuracy": [], "val_precision": [],
            "val_recall": [], "val_f1": [], "val_roc_auc": []
        }

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        self.model.eval()
        loader = loader or self.val_loader
        total_loss = 0.0
        y_true, y_prob = [], []
        
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits)
                
            total_loss += loss.item()
            y_true.extend(labels.cpu().squeeze().tolist() if labels.numel() > 1 else [labels.item()])
            y_prob.extend(probs.cpu().squeeze().tolist() if probs.numel() > 1 else [probs.item()])
            
        metrics = compute_metrics(y_true, y_prob)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    def fit(self) -> Dict[str, Any]:
        best_val_f1 = -1.0
        best_epoch = -1
        patience_counter = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
        
        print(f"--- Training {self.model_name} on {self.device} (AMP: {self.use_amp}) ---")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate()
            self.scheduler.step()
            
            val_f1 = val_metrics["f1"]
            val_auc = val_metrics["roc_auc"]
            val_acc = val_metrics["accuracy"]
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_acc)
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_f1)
            self.history["val_roc_auc"].append(val_auc)
            
            print(f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            
            # Checkpoint on best validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": val_metrics,
                }, best_checkpoint_path)
                print(f"  --> Checkpoint saved (New best F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch} (No improvement for {self.patience} epochs).")
                    break
                    
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
        print(f"Training completed. Best model saved at {best_checkpoint_path} (Epoch {best_epoch}, F1: {best_val_f1:.4f})")
        return {"best_f1": best_val_f1, "best_epoch": best_epoch, "history": self.history}
