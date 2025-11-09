# DeforNet

DeforNet is a lightweight convolutional neural network for satellite tile classification.  
Current scope: binary Trees vs Notrees detection to surface potential deforestation signals.  
Future scope: multi–land-cover classification and temporal change detection.

---

## Project Status

- Model architecture defined (DeforNet CNN).
- Dataset organized: `data/tiles/Trees`, `data/tiles/Notrees`.
- Training completed (best validation accuracy 0.9090 at epoch 13 of run shown below).
- Best checkpoint saved at: `outputs/notebook_run_1762682330/checkpoints/best.pt`.
- Test evaluation step available (ensure you actually ran it; add metrics below once done).

---

## Repository Structure

```
DeforNet/
├─ README.md
├─ MODEL_CARD.md
├─ requirements.txt
├─ src/ (optional if you migrated notebook code)
│  ├─ defornet.py
│  ├─ data.py
│  ├─ train.py
│  └─ utils.py
├─ notebooks/
│  └─ train_and_eval.ipynb
├─ outputs/
│  └─ notebook_run_<timestamp>/checkpoints/best.pt
└─ data/
   └─ tiles/
      ├─ Trees/
      └─ Notrees/
```

`data/` and `outputs/` should be git‑ignored.

---

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If you only trained in the notebook, ensure all needed packages are pinned in `requirements.txt`:

```
torch==2.1.2
torchvision==0.16.2
numpy==1.26.4
scikit-learn==1.3.2
tqdm==4.66.5
matplotlib==3.8.3
seaborn==0.13.2
PyYAML==6.0.1
```

(Adjust versions to those you actually used.)

---

## Dataset

Source (original multi-class dataset): [Trees in Satellite Imagery (Kaggle)](https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery)

For this binary prototype, only two folders are used:

```
data/tiles/
  Trees/
  Notrees/
```

Each must contain image files (jpg, png, etc.). No nested train/val/test folders are required—the code performs internal stratified splitting.

---

## Training (Notebook Workflow)

Open: `notebooks/train_and_eval.ipynb` and run cells top-to-bottom.  
Key hyperparameters defined near the top:

```
IMG_SIZE = 224
BATCH_SIZE = 64
LR = 0.0018
EPOCHS = 35
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
LABEL_SMOOTHING = 0.05
CHANNELS = (32, 64, 128, 256)
```

Early stopping patience = 7 epochs (triggered here at epoch 20).

Best validation accuracy reached: 0.9090 (epoch 13).  
Note: Validation accuracy oscillated ~±10%, indicating moderate variance (likely due to small validation set size and overfitting after epoch ~13).

---

## (Optional) Script-Based Training

If you migrated notebook code to `src/train.py`:

```bash
python src/train.py --config configs/defornet.yaml --data_dir data/tiles
```

Example `configs/defornet.yaml`:

```yaml
data_dir: "data/tiles"
output_dir: "outputs"
img_size: 224
epochs: 35
batch_size: 64
lr: 0.0018
weight_decay: 0.0001
val_split: 0.15
test_split: 0.15
seed: 42
num_workers: 4
device: "cuda"
label_smoothing: 0.05
warmup_epochs: 3
early_stop_patience: 7
amp: true
channels: [32, 64, 128, 256]
feature_dropout: 0.15
classifier_dropout: 0.25
```

---

## Evaluation & Metrics

After training, run the test evaluation cell (or script) to produce final metrics:

Recommended metrics to capture:
- Accuracy
- Precision / Recall / F1
- ROC AUC
- Confusion Matrix
- Number of test samples per class

Example snippet to add (if not yet in notebook):

```python
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix
model.eval()
probs_all = []
labels_all = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # adjust index if needed
        probs_all.extend(probs.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

preds_all = [1 if p >= 0.5 else 0 for p in probs_all]
precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="binary")
auc = roc_auc_score(labels_all, probs_all)
print(f"Test Accuracy: {sum(int(a==b) for a,b in zip(labels_all,preds_all))/len(labels_all):.4f}")
print(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} ROC AUC: {auc:.4f}")
```

Record final numbers in both `README.md` and `MODEL_CARD.md`.

---

## Saving & Loading the Model

Checkpoint dictionary contents (`best.pt`):
- `state_dict`
- `class_to_idx`
- `img_size`
- `channels`
- (optionally `epoch`, `val_acc`)

Rebuild & load:

```python
ckpt = torch.load("outputs/notebook_run_1762682330/checkpoints/best.pt", map_location="cpu")
model = DeforNet(
    num_classes=len(ckpt["class_to_idx"]),
    channels=ckpt["channels"],
    feature_dropout=0.15,
    classifier_dropout=0.25
)
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## Inference (Single Image)

Add a small `inference.py` (future task):

```bash
python inference.py --image path/to/image.jpg --checkpoint outputs/notebook_run_XXXX/checkpoints/best.pt
```

Outputs: predicted class + probability.

---

## Results Summary (Populate after test evaluation)

| Split | Metric | Value |
|-------|--------|-------|
| Validation (best) | Accuracy | 0.9090 |
| Test | Accuracy | (add) |
| Test | Precision | (add) |
| Test | Recall | (add) |
| Test | F1 | (add) |
| Test | ROC AUC | (add) |

Add a confidence interval for test accuracy once computed.

---

## Known Limitations

- Binary only; cannot differentiate other land-cover classes.
- Validation variance suggests need for larger validation set or stabilizing techniques (EMA weights, more data).
- Augmentation may not match real-world seasonal or sensor variation.

---

## Roadmap

1. Persist evaluation artifacts automatically (classification report, confusion matrix JSON).
2. Add inference script + batch folder prediction.
3. Expand to multi-class (Forest, Water, Urban, Agriculture, Bare).
4. Introduce temporal change detection (Trees→Notrees sequence per tile).
5. Add basic tests (`tests/`) + CI workflow.
6. Add MODEL_CARD.md and LICENSE (MIT or Apache 2.0).
7. Optional: Export ONNX and provide FastAPI microservice.

---

## License

(Choose one; example MIT)

```
MIT License
Copyright (c) 2025 ...
```

Add a `LICENSE` file to formalize this.

---

## Citation / Attribution

If you publish or release, cite dataset source and credit the Kaggle dataset author.

---

## Contributing

Open PRs for:
- New data sources
- Additional metrics
- Change detection module

Run tests & lint (to be added) before submitting.

---

## Acknowledgements

- Dataset: Kaggle Trees in Satellite Imagery.
- Inspiration from lightweight CNN design patterns (two conv blocks + pooling pyramid).

---

## Quick Reproduce Checklist

1. Create `data/tiles/Trees` & `data/tiles/Notrees` folders with images.
2. Install dependencies.
3. Run notebook (or script).
4. Save best checkpoint.
5. Evaluate test metrics.
6. Populate README and MODEL_CARD with results.

---

## FAQ

Q: Why does validation accuracy jump around?  
A: Small validation set + overfitting after early epochs + stochastic training. Consider larger val split or stabilizing (EMA, lower LR, reduced augmentation).

Q: Why is top-3 accuracy removed?  
A: Binary classification has only 2 classes; `k=3` is invalid.

Q: Do I need `__init__.py` in `data/`?  
A: No. Only code packages need `__init__.py`; data folders are read directly by `ImageFolder`.

---

