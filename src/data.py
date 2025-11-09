from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List, Dict, Tuple
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_transforms(img_size: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.15),
        transforms.RandomRotation(25),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_ = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, eval_

def stratified_indices(targets: List[int], val_split: float, test_split: float, seed: int):
    indices = list(range(len(targets)))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    tv_idx, test_idx = next(sss1.split(indices, [targets[i] for i in indices]))
    tv_indices = [indices[i] for i in tv_idx]
    test_indices = [indices[i] for i in test_idx]
    y_tv = [targets[i] for i in tv_indices]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split/(1-test_split), random_state=seed)
    train_rel, val_rel = next(sss2.split(tv_indices, y_tv))
    train_indices = [tv_indices[i] for i in train_rel]
    val_indices = [tv_indices[i] for i in val_rel]
    return train_indices, val_indices, test_indices

def get_loaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    val_split: float,
    test_split: float,
    seed: int,
    num_workers: int,
):
    train_tfms, eval_tfms = make_transforms(img_size)
    base = datasets.ImageFolder(data_dir)
    targets = [s[1] for s in base.samples]
    train_idx, val_idx, test_idx = stratified_indices(targets, val_split, test_split, seed)

    train_ds = datasets.ImageFolder(data_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(data_dir, transform=eval_tfms)
    test_ds  = datasets.ImageFolder(data_dir, transform=eval_tfms)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.class_to_idx