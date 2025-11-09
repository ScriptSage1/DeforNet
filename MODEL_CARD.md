# Model Card: DeforNet (Binary Trees vs Notrees)

## Overview

DeforNet is a lightweight convolutional neural network designed to classify satellite image tiles as containing tree cover (`Trees`) or lacking tree cover (`Notrees`). The goal is to provide an initial signal for potential deforestation or land-cover change monitoring.

## Intended Use

- Rapid screening of tiles to flag possible tree loss areas.
- Pre-filtering before more detailed, higher-resolution or multi-temporal analysis.
- Educational / prototyping demonstration of a custom CNN for remote sensing.

Not intended for:
- High-stakes decision making without human review.
- Fine-grained ecological assessment.
- Direct biomass or carbon stock estimation.

## Users

- Researchers experimenting with deforestation detection pipelines.
- Developers building a multi-stage land-cover monitoring system.
- Students learning applied computer vision in environmental contexts.

## Data

- Source dataset: [Trees in Satellite Imagery (Kaggle)](https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery)
- Classes used in this prototype: `Trees`, `Notrees`
- Internal split (stratified): Train ~70%, Validation 15%, Test 15% (adjustable via config).
- Image size: Center/Random crops resized to 224×224 for training.
- Augmentations: RandomResizedCrop, flips (H/V), ColorJitter, Rotation (±25°), normalization with ImageNet statistics.


## Model Architecture

DeforNet (baseline):
- Input: 3×224×224
- Convolutional stages: channel progression `(32, 64, 128, 256)`
- Each stage: two conv blocks (Conv → BatchNorm → ReLU → optional Dropout) + MaxPool
- Global Average Pooling
- Fully-connected classifier head with dropout
- Parameter count: 1.17M

Initialization:
- Kaiming Normal for conv layers
- Xavier Uniform for linear layers
- BatchNorm weights = 1, biases = 0

Regularization:
- Feature dropout (0.15)
- Classifier dropout (0.25)
- Label smoothing (0.05)

Scheduler:
- Warmup (3 epochs) → Cosine decay to min LR (1e-6)

Optimizer:
- AdamW (lr=0.0018, weight_decay=1e-4)

## Training Setup

- Hardware: (google collab gpu)
- Mixed Precision: Enabled (PyTorch AMP)
- Epochs: 35 (early stopped at epoch 20 due to patience=7)
- Seed: 42

## Metrics

Validation (best epoch):
- Accuracy: 0.9090 (epoch 13)


### Variance / Stability

Validation accuracy oscillated ± ~10% after peak, indicating moderate variance due to small validation set size and overfitting beyond epoch ~13. Early stopping captured a near‑optimal checkpoint.

## Thresholds

Current predictions use argmax of logits (probability threshold 0.5). For operational deployment:
- Evaluate precision/recall trade-offs by sweeping thresholds.
- Potentially prioritize recall (minimize missed tree cover loss) depending on monitoring goals.

## Ethical & Societal Considerations

- False negatives (missed tree loss) can delay detection.
- False positives may trigger unnecessary follow-up or misallocate resources.
- Geographic bias: Model may generalize poorly to regions unlike the training set (different vegetation, seasonality, sensor characteristics).
- Regular audits and human validation recommended before decision-making.

## Limitations

- Binary only; cannot distinguish agriculture, urban, water, or regrowth.
- No temporal modeling yet; a single tile may not indicate active deforestation.
- Augmentation may not cover full real-world variability (e.g., haze, seasonal snow).
- Potential class imbalance not explicitly managed beyond label smoothing.

## Potential Improvements

1. Multi-class expansion (Forest, Water, Urban, Agriculture, Bare).
2. Temporal change detection module comparing sequential tiles.
3. Exponential Moving Average (EMA) weights for smoother generalization.
4. Advanced augmentation: Mixup / CutMix / random erasing.
5. Improved metrics: MCC (Matthews Correlation Coefficient), calibration curves (ECE).
6. Semi-supervised or active learning strategies for new unlabeled regions.
7. Lightweight deployment: ONNX export + FastAPI inference endpoint.

## Model Use Guidelines

- Always inspect probability outputs; consider thresholds tuned on validation ROC/PR curves.
- Recalibrate if dataset distribution shifts (new satellite source or region).
- Re-train or fine-tune for multi-class before claiming broader land-cover capabilities.

## Security & Privacy

- No personally identifiable information in dataset.
- Satellite imagery may have regulatory constraints depending on region; ensure compliance if expanding dataset sources.

## How to Reproduce

1. Prepare `data/tiles/Trees` and `data/tiles/Notrees`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run notebook `train_and_eval.ipynb`.
4. Save best checkpoint.
5. Execute test evaluation cell and update metrics here.
6. Optionally export ONNX:
   ```python
   dummy = torch.randn(1, 3, 224, 224).to(device)
   torch.onnx.export(model, dummy, "outputs/defornet_binary.onnx", opset_version=17)
   ```

## Artifacts

| Artifact | Path |
|----------|------|
| Best Checkpoint | `outputs/notebook_run_1762682330/checkpoints/best.pt` |
| Notebook | `notebooks/train_and_eval.ipynb` |
