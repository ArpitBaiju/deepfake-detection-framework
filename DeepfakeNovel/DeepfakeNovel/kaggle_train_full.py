"""
DAF-Net Full Scale Kaggle Standalone Training Script.
Ready to run in a Kaggle Notebook code cell or as a standalone Kaggle script.
"""
# ==============================================================================
# CELL 1: SETUP, SEEDING & GPU ACCELERATION
# ==============================================================================
import os
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")
if torch.cuda.is_available():
    print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# CELL 2: COMPLETE 140k DATASET DISCOVERY
# ==============================================================================
def find_split_dirs(base_path="/kaggle/input"):
    splits = {
        "train": {"real": [], "fake": []},
        "val":   {"real": [], "fake": []},
        "test":  {"real": [], "fake": []}
    }
    print(f"Scanning dataset folders under {base_path} ...")
    for root, dirs, files in os.walk(base_path):
        r_lower = root.lower().replace("\\", "/")
        b_name = os.path.basename(root).lower()
        is_real = b_name in ["real", "0_real"] or ("real" in b_name and "fake" not in b_name)
        is_fake = b_name in ["fake", "1_fake"] or "fake" in b_name
        
        if is_real or is_fake:
            target = "real" if is_real else "fake"
            if "valid" in r_lower or "val" in r_lower:
                splits["val"][target].append(root)
            elif "test" in r_lower:
                splits["test"][target].append(root)
            else:
                splits["train"][target].append(root)
                
    print("\nDiscovered Dataset Folders:")
    for s, data in splits.items():
        print(f"  [{s.upper()}] Real dirs: {len(data['real'])} | Fake dirs: {len(data['fake'])}")
    return splits

SPLITS = find_split_dirs()


# ==============================================================================
# CELL 3: FAST 140k DATASET LOADER & AUGMENTATIONS
# ==============================================================================
class FullDeepfakeDataset(Dataset):
    def __init__(self, real_dirs: List[str], fake_dirs: List[str], transform=None, name="dataset"):
        self.samples = []  # List of (filepath, label: 0=Real, 1=Fake)
        self.transform = transform
        self.name = name
        valid_exts = (".jpg", ".jpeg", ".png")
        
        print(f"Indexing {name} images ...")
        for d in real_dirs:
            for entry in os.scandir(d):
                if entry.is_file() and entry.name.lower().endswith(valid_exts):
                    self.samples.append((entry.path, 0))
        real_count = len(self.samples)
        
        for d in fake_dirs:
            for entry in os.scandir(d):
                if entry.is_file() and entry.name.lower().endswith(valid_exts):
                    self.samples.append((entry.path, 1))
        fake_count = len(self.samples) - real_count
        print(f"-> [{name.upper()}] Total: {len(self.samples):,} images | Real: {real_count:,} | Fake: {fake_count:,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 4

# Rich facial artifact data augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = FullDeepfakeDataset(SPLITS["train"]["real"], SPLITS["train"]["fake"], transform=train_transform, name="Train")

if len(SPLITS["val"]["real"]) > 0 and len(SPLITS["val"]["fake"]) > 0:
    val_dataset = FullDeepfakeDataset(SPLITS["val"]["real"], SPLITS["val"]["fake"], transform=val_transform, name="Validation")
else:
    print("Partitioning 85% train / 15% validation...")
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
print(f"\n[DATALOADERS READY]: {len(train_loader):,} train batches/epoch | {len(val_loader):,} val batches/epoch")


# ==============================================================================
# CELL 4: DAF-NET ARCHITECTURE (V1)
# ==============================================================================
class DownsampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ArtifactBlockV1(nn.Module):
    def __init__(self, c, init_alpha=1.0):
        super().__init__()
        self.local_branch = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )
        self.gate_conv = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x, return_artifacts=False):
        f_loc = self.local_branch(x)
        f_ctx = self.context_branch(x)
        dev = torch.abs(f_loc - f_ctx)
        gate = self.gate_conv(dev)
        out = x + self.alpha * (x * gate)
        if return_artifacts:
            return out, {"deviation": dev, "gate": gate, "alpha": self.alpha.item()}
        return out

class DAFNetV1(nn.Module):
    def __init__(self, in_c=3, stem_c=32, stage_c=(64, 128, 256), dropout=0.2, init_alpha=1.0):
        super().__init__()
        c1, c2, c3 = stage_c
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, stem_c, 3, 1, 1, bias=False), nn.BatchNorm2d(stem_c), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(stem_c, c1, 3, 1, 1, bias=False), nn.BatchNorm2d(c1), nn.LeakyReLU(0.1, inplace=True)
        )
        self.ab1 = ArtifactBlockV1(c1, init_alpha)
        self.ds1 = DownsampleBlock(c1, c2)
        self.ab2 = ArtifactBlockV1(c2, init_alpha)
        self.ds2 = DownsampleBlock(c2, c3)
        self.ab3 = ArtifactBlockV1(c3, init_alpha)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, 1)

    def forward(self, x, return_artifacts=False):
        feat = self.stem(x)
        if return_artifacts:
            feat, art1 = self.ab1(feat, True)
            feat = self.ds1(feat)
            feat, art2 = self.ab2(feat, True)
            feat = self.ds2(feat)
            feat, art3 = self.ab3(feat, True)
            logit = self.fc(self.dropout(self.gap(feat).flatten(1)))
            return logit, {"block1": art1, "block2": art2, "block3": art3}
        else:
            feat = self.ab1(feat)
            feat = self.ds1(feat)
            feat = self.ab2(feat)
            feat = self.ds2(feat)
            feat = self.ab3(feat)
            return self.fc(self.dropout(self.gap(feat).flatten(1)))

model_v1 = DAFNetV1().to(device)
print(f"DAF-Net V1 Model instantiated! Total Parameters: {sum(p.numel() for p in model_v1.parameters()):,}")


# ==============================================================================
# CELL 5: 12-EPOCH TRAINING ENGINE WITH EARLY STOPPING & CHECKPOINTING
# ==============================================================================
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
            total_loss += loss.item()
            y_true.extend(labels.cpu().squeeze().tolist())
            y_prob.extend(probs.cpu().squeeze().tolist())
            
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "cm": confusion_matrix(y_true, y_pred).tolist()
    }

def train_full_pipeline(model, train_loader, val_loader, epochs=12, lr=1e-4, wd=1e-4, patience=4, model_name="dafnet_v1"):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    best_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    total_batches = len(train_loader)
    
    history = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_precision": [], "val_recall": [],
        "val_f1": [], "val_roc_auc": []
    }
    
    print(f"\n======================================================")
    print(f"Starting Full 12-Epoch Training: {model_name} on {device}")
    print(f"Total Batches/Epoch: {total_batches:,} | Early Stopping Patience: {patience}")
    print(f"======================================================")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if step % 250 == 0 or step == total_batches:
                elapsed = time.time() - epoch_start
                avg_loss = epoch_loss / step
                speed = step / elapsed
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Step {step:04d}/{total_batches:04d} | Running Loss: {avg_loss:.4f} | {speed:.1f} b/s")
                
        print(f"Evaluating epoch {epoch:02d} on validation set ...")
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        
        train_loss = epoch_loss / total_batches
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]
        val_auc = val_metrics["roc_auc"]
        val_acc = val_metrics["accuracy"]
        epoch_duration = time.time() - epoch_start
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_f1)
        history["val_roc_auc"].append(val_auc)
        
        print(f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] Completed in {epoch_duration/60:.1f}m:")
        print(f"    Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"    Val Acc:    {val_acc:.4f} | Val F1: {val_f1:.4f} | ROC-AUC: {val_auc:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"    Confusion Matrix: {val_metrics['cm']}\n")
        
        # Checkpoint based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_metrics,
                "history": history
            }, f"{model_name}_best.pth")
            print(f"  --> [CHECKPOINT SAVED]: Best validation F1 ({best_f1:.4f}) at epoch {epoch}.\n")
        else:
            patience_counter += 1
            print(f"  --> No improvement in F1 for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING]: Stopping training early at epoch {epoch}. Best validation F1 was {best_f1:.4f} at epoch {best_epoch}.")
                break
                
    with open(f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return history

# Run training
history_v1 = train_full_pipeline(model_v1, train_loader, val_loader, epochs=12, lr=1e-4, patience=4, model_name="dafnet_v1")


# ==============================================================================
# CELL 6: EVALUATION CURVES & ARTIFACT VISUALIZATION
# ==============================================================================
epochs_range = range(1, len(history_v1["train_loss"]) + 1)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, history_v1["train_loss"], "b-o", label="Train Loss")
plt.plot(epochs_range, history_v1["val_loss"], "r-o", label="Val Loss")
plt.title("Train vs Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history_v1["val_accuracy"], "g-o", label="Accuracy")
plt.plot(epochs_range, history_v1["val_f1"], "m-o", label="F1-Score")
plt.title("Validation Accuracy & F1-Score")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history_v1["val_roc_auc"], "c-o", label="ROC-AUC")
plt.title("Validation ROC-AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("dafnet_v1_training_curves.png", dpi=200)
plt.show()

# Artifact map visualization
model_v1.eval()
sample_imgs, sample_labels = next(iter(val_loader))
sample_imgs = sample_imgs.to(device)

with torch.no_grad():
    logits, art_maps = model_v1(sample_imgs[:4], return_artifacts=True)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
for row in range(4):
    img = sample_imgs[row].cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    gt = "Fake" if sample_labels[row].item() == 1 else "Real"
    pred_str = "Fake" if probs[row] >= 0.5 else "Real"
    
    axes[row, 0].imshow(img)
    axes[row, 0].set_title(f"GT: {gt} | Pred: {pred_str} ({probs[row]:.2f})")
    axes[row, 0].axis("off")
    
    for col, blk in enumerate(["block1", "block2", "block3"], start=1):
        gate = art_maps[blk]["gate"][row].mean(dim=0).cpu().numpy()
        axes[row, col].imshow(gate, cmap="inferno")
        axes[row, col].set_title(f"{blk.upper()} Artifact Gate")
        axes[row, col].axis("off")
        
plt.tight_layout()
plt.savefig("artifact_map_visualizations.png", dpi=200)
plt.show()
