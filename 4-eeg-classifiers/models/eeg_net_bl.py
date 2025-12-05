# models/eegnet_lite.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetBL(nn.Module):
    """
    Baseline EEG classifier inspired by classic EEGNet:
      - Temporal convolution (learn frequency filters)
      - Depthwise spatial convolution (per-channel decomposition)
      - Separable convolution (lightweight feature extraction)
      - Global average pooling
      - Linear classifier

    Input:  [B, C=122, T]
    Output: [B, num_classes]
    """

    def __init__(
        self,
        num_channels: int = 122,
        num_classes: int = 20,
        temporal_filters: int = 8,   # F1
        temporal_kernel: int = 64,   # K
        depth_multiplier: int = 2,   # D
        dropout: float = 0.25,
    ):
        super().__init__()

        C = num_channels
        F1 = temporal_filters
        D = depth_multiplier
        K = temporal_kernel
        pad = K // 2

        # 1) Temporal convolution (frequency filtering)
        self.temporal = nn.Sequential(
            nn.Conv1d(C, F1, kernel_size=K, padding=pad, bias=False),
            nn.BatchNorm1d(F1),
            nn.ELU()
        )

        # 2) Depthwise spatial convolution (per-channel mixing)
        self.depthwise = nn.Sequential(
            nn.Conv1d(F1, F1 * D, kernel_size=1, groups=F1, bias=False),
            nn.BatchNorm1d(F1 * D),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(dropout),
        )

        # 3) Separable convolution (depthwise temporal + pointwise)
        self.separable = nn.Sequential(
            nn.Conv1d(F1 * D, F1 * D, kernel_size=16, padding=8, groups=F1 * D, bias=False),
            nn.Conv1d(F1 * D, F1 * D, kernel_size=1, bias=False),
            nn.BatchNorm1d(F1 * D),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=8),
            nn.Dropout(dropout),
        )

        # 4) Classifier
        self.classifier = nn.Linear(F1 * D, num_classes)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, C, T] or [B, T, C]
        """
        # Ensure channel dimension is in dim=1
        if x.shape[1] != 122:
            x = x.transpose(1, 2)

        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)

        # Global average pooling
        x = x.mean(dim=-1)   # [B, features]

        return self.classifier(x)