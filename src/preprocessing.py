"""
Preprocessing and Dataset Pipeline for Crop Health Monitoring System.

Features:
- Deterministic stratified train/val/test split preventing data leakage.
- Strong data augmentation pipeline for model generalization (train split only).
- Standardized ImageNet normalization and resizing (224x224).
- PyTorch Dataset and DataLoader generators.
"""

import os
import shutil
import random
import json
from pathlib import Path
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Standard ImageNet normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)


def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Returns data transforms for train, validation, and testing sets.
    Augmentation is strictly isolated to the training split.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


def split_dataset(
    raw_dir: str,
    output_base_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Dict[str, Dict[str, int]]:
    """
    Performs stratified deterministic dataset splitting into train, val, and test subsets.
    Physically organizes images into dataset/train, dataset/val, dataset/test.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    set_seed(seed)

    raw_path = Path(raw_dir)
    base_path = Path(output_base_dir)

    train_path = base_path / "train"
    val_path = base_path / "val"
    test_path = base_path / "test"

    for p in [train_path, val_path, test_path]:
        p.mkdir(parents=True, exist_ok=True)
        # Clear existing files inside rather than removing directory itself
        for item in p.glob("**/*"):
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass

    classes = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])
    split_summary = {"train": {}, "val": {}, "test": {}}

    for cls in classes:
        cls_raw_dir = raw_path / cls
        images = sorted([f for f in cls_raw_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # Remaining goes to test to ensure total count preservation
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for subset_name, subset_imgs, dest_dir in [
            ("train", train_imgs, train_path),
            ("val", val_imgs, val_path),
            ("test", test_imgs, test_path),
        ]:
            subset_cls_dir = dest_dir / cls
            subset_cls_dir.mkdir(parents=True, exist_ok=True)
            for img_file in subset_imgs:
                shutil.copy2(img_file, subset_cls_dir / img_file.name)
            split_summary[subset_name][cls] = len(subset_imgs)

    print("\nDataset Stratified Split Completed:")
    for subset in ["train", "val", "test"]:
        total = sum(split_summary[subset].values())
        print(f"  {subset.upper()} split: {total} total images across {len(classes)} classes")
        for cls, count in split_summary[subset].items():
            print(f"    - {cls}: {count}")

    return split_summary


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """
    Builds PyTorch DataLoaders for train, val, and test sets.
    Saves class mappings into models/class_names.json.
    """
    transforms_dict = get_transforms()
    base_path = Path(dataset_dir)

    image_datasets = {
        split: datasets.ImageFolder(
            root=str(base_path / split),
            transform=transforms_dict[split]
        )
        for split in ["train", "val", "test"]
    }

    class_names = image_datasets["train"].classes

    # Ensure class mapping directory exists
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        ),
    }

    return dataloaders, class_names


def preprocess_single_image(img_input, target_size=IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocesses a single image for inference.
    Accepts PIL Image, image file path, or numpy array.
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        img = Image.fromarray(img_input).convert("RGB")

    eval_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    tensor = eval_transform(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    return tensor


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "dataset" / "raw"
    dataset_dir = project_root / "dataset"

    print("Executing dataset split...")
    summary = split_dataset(str(raw_dir), str(dataset_dir))

    print("\nVerifying DataLoader generation...")
    loaders, classes = create_dataloaders(str(dataset_dir), batch_size=16)
    print(f"DataLoaders successfully instantiated for classes: {classes}")
    for split, loader in loaders.items():
        print(f"  {split} DataLoader: {len(loader.dataset)} items, {len(loader)} batches")
