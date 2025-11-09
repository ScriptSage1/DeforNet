import torch
import torch.nn as nn
from typing import Sequence, Tuple, List

def conv_block(in_ch: int, out_ch: int, k: int = 3, p: int = 1, dropout: float = 0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class DeforNet(nn.Module):
    """
    Lightweight CNN for tile classification (from scratch).
    Uses repeated conv blocks + max pooling + GAP + linear classifier.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        feature_dropout: float = 0.15,
        classifier_dropout: float = 0.25,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        prev = in_channels
        for ch in channels:
            blocks.append(conv_block(prev, ch, dropout=feature_dropout))
            blocks.append(conv_block(ch, ch, dropout=feature_dropout))
            blocks.append(nn.MaxPool2d(2))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(classifier_dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def build_model(
    num_classes: int,
    img_size: int = 224,
    channels: Sequence[int] = (32, 64, 128, 256),
    feature_dropout: float = 0.15,
    classifier_dropout: float = 0.25,
) -> Tuple[nn.Module, int]:
    model = DeforNet(
        num_classes=num_classes,
        channels=channels,
        feature_dropout=feature_dropout,
        classifier_dropout=classifier_dropout,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable