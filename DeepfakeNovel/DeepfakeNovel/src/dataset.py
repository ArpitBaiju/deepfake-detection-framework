"""
PyTorch Dataset and DataLoader preparation for manjilkarki/deepfake-and-real-images.
Handles Real vs Fake folder structure with robust checks for class balance.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, Dict


class DeepfakeDataset(Dataset):
    """
    Standard dataset loader compatible with Kaggle's manjilkarki/deepfake-and-real-images dataset.
    Structure:
      root_dir/
        Real/ (or real/)
        Fake/ (or fake/)
    """
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (filepath, label: 0=real, 1=fake)
        self.class_counts = {0: 0, 1: 0}
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
            
        real_dirs = [d for d in os.listdir(root_dir) if "real" in d.lower()]
        fake_dirs = [d for d in os.listdir(root_dir) if "fake" in d.lower()]
        
        if not real_dirs or not fake_dirs:
            # Check nested subdirectories (e.g., train/real or dataset/train/real)
            for root, dirs, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(root, f)
                        parent = os.path.basename(os.path.dirname(path)).lower()
                        if "real" in parent:
                            self.samples.append((path, 0))
                            self.class_counts[0] += 1
                        elif "fake" in parent:
                            self.samples.append((path, 1))
                            self.class_counts[1] += 1
        else:
            for r_dir in real_dirs:
                p = os.path.join(root_dir, r_dir)
                for f in os.listdir(p):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(p, f), 0))
                        self.class_counts[0] += 1
                        
            for f_dir in fake_dirs:
                p = os.path.join(root_dir, f_dir)
                for f in os.listdir(p):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(p, f), 1))
                        self.class_counts[1] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    def get_class_weights(self) -> Optional[torch.Tensor]:
        total = len(self.samples)
        if total == 0 or self.class_counts[0] == 0 or self.class_counts[1] == 0:
            return None
        # pos_weight = count(negatives) / count(positives)
        # Real=0 (neg), Fake=1 (pos)
        pos_weight = self.class_counts[0] / self.class_counts[1]
        return torch.tensor([pos_weight], dtype=torch.float32)


def get_transforms(image_size: int = 128) -> Dict[str, transforms.Compose]:
    """
    Standard transforms.
    Train: Resize, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
    Val/Test: Resize, ToTensor, Normalize
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {"train": train_transform, "val": val_transform}
