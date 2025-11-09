import os
import random
import numpy as np
import torch
import yaml
from typing import Any, Dict

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_checkpoint(state, path: str):
    torch.save(state, path)

def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            frac = (epoch + 1) / self.warmup_epochs
            for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base * frac
        else:
            t = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cos = 0.5 * (1 + np.cos(np.pi * t))
            for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = self.min_lr + (base - self.min_lr) * cos