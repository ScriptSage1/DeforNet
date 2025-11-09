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
`requirements.txt`:

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

---

## Dataset

Source (original multi-class dataset): [Trees in Satellite Imagery (Kaggle)](https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery)

For this binary prototype, only two folders are used:

```
data/tiles/
  Trees/
  Notrees/
```

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

## Known Limitations

- Binary only; cannot differentiate other land-cover classes.
- Validation variance suggests need for larger validation set or stabilizing techniques (EMA weights, more data).
- Augmentation may not match real-world seasonal or sensor variation.

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
---

