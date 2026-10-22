"""Small 1D convolutional network for 8 x 25 EMG windows."""

from __future__ import annotations

import torch
from torch import nn

from .types import CHANNEL_COUNT


class SmallEmgCnn(nn.Module):
    """A deliberately small baseline to limit overfitting on early datasets."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(CHANNEL_COUNT, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(32, class_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))

