import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import json
from typing import List, Dict

def label_smoothing_ce(logits, targets, smoothing: float, num_classes: int):
    if smoothing <= 0:
        return F.cross_entropy(logits, targets)
    with torch.no_grad():
        dist = torch.zeros_like(logits)
        dist.fill_(smoothing / (num_classes - 1))
        dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
    log_probs = F.log_softmax(logits, dim=1)
    return -(dist * log_probs).sum(dim=1).mean()

def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        match = (topk == targets.unsqueeze(1)).any(dim=1)
        return match.float().mean().item()

def write_report(labels: List[int], preds: List[int], mapping: Dict[int, str], out_dir: str):
    names = [mapping[i] for i in range(len(mapping))]
    report = classification_report(labels, preds, target_names=names, digits=4)
    cm = confusion_matrix(labels, preds).tolist()
    with open(f"{out_dir}/classification_report.txt", "w") as f:
        f.write(report)
    with open(f"{out_dir}/confusion_matrix.json", "w") as f:
        json.dump(cm, f, indent=2)
    return report