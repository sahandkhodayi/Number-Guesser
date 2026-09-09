"""PyTorch CNN model used to train Number Guesser."""

import torch
from torch import nn


class _MainModel(nn.Module):
    """Small CNN matching the C inference implementation exactly."""

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        # CHANGE: kept the trained architecture, but removed placeholder/debug noise.
        # Shape flow for MNIST:
        # [B, 1, 28, 28] -> [B, 32, 14, 14] -> [B, 32, 7, 7] -> [B, 1568] -> [B, 10]
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 32 channels * 7 height * 7 width = 1568 features.
            nn.Linear(hidden_units * 7 * 7, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        return self.classifier(x)
