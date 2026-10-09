"""
1. downloading dataset
2. splitting into training, validation and testing datasets
3. applying image transformations
4. creating DataLoader
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import kagglehub
from PIL import Image
from torchvision import transforms

# get constants
from .config import (
    CLASS_DIRS, CLASSES, 
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, 
    KAGGLE_DATASET, 
    SPLITS_CSV
)


def get_dataset() -> Path:
    """
    download dataset (cached after first call)
    """
    return Path(kagglehub.dataset_download(KAGGLE_DATASET)) / 'COVID-19_Radiography_Dataset'


def make_splits(out=SPLITS_CSV, val=0.1, test=0.1, seed=42):
    """
    split into testing, validation and training sets and write splits to csv
    """
    root = get_dataset() # get dataset path
    rng = random.Random(seed)
    rows = []

    for c in CLASSES:
        # get and shuffle all images belonging to the class
        img_dir = root / CLASS_DIRS[c] / 'images' # get image directory for each class relative to dataset root
        files = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() == ".png")
        rng.shuffle(files)

        # split images in 1:1:8 ratio for testing:validation:training
        n_test = round(len(files) * test)
        n_val = round(len(files) * val)
        for i, name in enumerate(files):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            rows.append({"path": f"{CLASS_DIRS[c]}/images/{name}", "label": c, "split": split}) # store each image's path, class and split

    # write the rows containing image path, class and split to csv
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def split_counts(splits_csv=SPLITS_CSV):
    """
    return count of each split for each class ({split: {class: count}})
    """
    with open(splits_csv, newline="") as f:
        counts = Counter((r["split"], r["label"]) for r in csv.DictReader(f))
    return {s: {c: counts[(s, c)] for c in CLASSES} for s in ["train", "val", "test"]}


def get_transform(train: bool):
    """
    turn images into tensors
    (no horizontal flip because fixed anatomy of heart on left)
    """
    # apply random distortion to training images
    augment = [
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), # randomly rotate, shift and scale image within constraints
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # change brightness and contrast by up to +/-20%
    ] if train else []

    # resize images, turn into tensors then normalise for scaling for resnet18
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        *augment,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(image: torch.Tensor) -> np.ndarray:
    """
    turn normalised tensor back into image for plotting
    """
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(image, 0.0, 1.0)


class ChestXRayDataset(torch.utils.data.Dataset):
    """
    custom dataset with fixed split as done in splits csv
    """
    def __init__(self, split, transform=None, root=None, splits_csv=SPLITS_CSV):
        self.root = Path(root) if root else get_dataset()
        self.transform = transform or get_transform(train=False)
        with open(splits_csv, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.paths = [r["path"] for r in rows]
        self.labels = [CLASSES.index(r['label']) for r in rows]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.root / self.paths[index]).convert("RGB")
        return self.transform(image), self.labels[index]


def make_loaders(batch_size=32, num_workers=4, samples_per_epoch=None):
    """
    build training, testing and validation loaders
    """
    root = get_dataset()

    # get created datasets
    train_ds = ChestXRayDataset('train', get_transform(train=True), root)
    val_ds = ChestXRayDataset('val', get_transform(train=False), root)
    test_ds = ChestXRayDataset('test', get_transform(train=False), root)

    # using class-balanced sampler for training to combat class imbalance
    class_counts = np.bincount(train_ds.labels, minlength=len(CLASSES)) # number of images in each class in training dataset
    weights = 1.0 / class_counts[train_ds.labels] # weight inversely proportional to number of images in class
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, 
        num_samples=samples_per_epoch or len(train_ds), 
        replacement=True
    )

    kwargs = dict(
        batch_size=batch_size, 
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    # return dataloaders for each dataset
    return (torch.utils.data.DataLoader(train_ds, sampler=sampler, **kwargs),
            torch.utils.data.DataLoader(val_ds, shuffle=False, **kwargs),
            torch.utils.data.DataLoader(test_ds, shuffle=False, **kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset and write training/validation/testing split')
    parser.add_argument('--val', type=float, default=0.1, help='fraction of images in each class assigned to validation')
    parser.add_argument('--test', type=float, default=0.1, help='fraction of images in each class assigned to testing')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    path = make_splits(val=args.val, test=args.test, seed=args.seed) # write to splits csv
    print(f'Wrote to {path}')
    for split, counts in split_counts(path).items():
        print(f'  {split:5s}  ' + '  '.join(f'{c}={n}' for c, n in counts.items()))
