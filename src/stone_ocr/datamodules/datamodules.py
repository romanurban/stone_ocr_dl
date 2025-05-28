from pathlib import Path
from typing import List, Tuple
import json, random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
from collections import Counter
import torch

IMG_SIZE = 256 

def make_transforms(train=True):
    if train:
        aug = [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    else:
        aug = [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
    return A.Compose(aug)

class FolderDataset(Dataset):
    def __init__(self, root: Path, items: List[Tuple[str, int]], class_names, train=True):
        self.root = root
        self.items = items
        self.class_names = class_names
        self.tf = make_transforms(train)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rel_path, label = self.items[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        img = self.tf(image=np.array(img))["image"]
        return img, torch.tensor(label)

class DefectDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=16):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.n_classes = len(self.class_names)

    def setup(self, stage=None):
        split_file = self.data_dir / "splits.json"
        if not split_file.exists():
            self._make_splits(split_file)
        splits = json.load(open(split_file))
        self.train_set = FolderDataset(self.data_dir, splits["train"], self.class_names, train=True)
        self.val_set   = FolderDataset(self.data_dir, splits["val"], self.class_names, train=False)
        self.test_set  = FolderDataset(self.data_dir, splits["test"], self.class_names, train=False)

    def train_dataloader(self): return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    def val_dataloader(self):   return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=8, persistent_workers=True)
    def test_dataloader(self):  return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=8, persistent_workers=True)

    def _make_splits(self, path):
        all_items = []
        for lbl, cname in enumerate(self.class_names):
            for img in (self.data_dir / cname).glob("*"):
                if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                    continue
                all_items.append((str(img.relative_to(self.data_dir)), lbl))


        random.shuffle(all_items)
        n = len(all_items)
        splits = {
            "train": all_items[:int(0.7*n)],
            "val":   all_items[int(0.7*n):int(0.85*n)],
            "test":  all_items[int(0.85*n):]
        }
        json.dump(splits, open(path, "w"), indent=2)

    def compute_class_weights(self):
        """Computes class weights from training labels."""
        label_counts = Counter()
        for _, label in self.train_set:
            label_counts[int(label)] += 1

        total = sum(label_counts.values())
        weights = [total / label_counts[i] for i in range(self.n_classes)]
        weights = torch.tensor(weights, dtype=torch.float)
        return weights