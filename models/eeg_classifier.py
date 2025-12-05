# /jet/home/bermudez/exploring-eeg/models/eegnet_baseline.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetBaseline(nn.Module):
    """
    Minimal EEGNet-like baseline for 20-class classification.
    Expects input as [B, C, T]. We'll reshape to [B, 1, C, T] inside forward.
    """
    def __init__(self, num_channels=122, num_classes=20, input_time=500,
                 F1=8, D=2, F2=None, dropout=0.25):
        super().__init__()
        if F2 is None:
            F2 = F1 * D  # common EEGNet setting

        self.num_channels = num_channels
        self.input_time = input_time

        # Block 1: temporal convolution
        self.conv_time = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, 16), padding=(0, 8), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: depthwise spatial convolution
        self.depthwise = nn.Conv2d(
            in_channels=F1, out_channels=F1*D,
            kernel_size=(num_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop2 = nn.Dropout(dropout)

        # Block 3: separable convolution (depthwise-temporal + pointwise)
        self.separable_depth = nn.Conv2d(
            in_channels=F1*D, out_channels=F1*D,
            kernel_size=(1, 16), padding=(0, 8),
            groups=F1*D, bias=False
        )
        self.separable_point = nn.Conv2d(
            in_channels=F1*D, out_channels=F2, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop3 = nn.Dropout(dropout)

        # classifier
        # compute feature size after pools for linear layer
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, input_time)
            h = self._features(dummy)
            feat_dim = h.shape[1] * h.shape[2] * h.shape[3]
        self.fc = nn.Linear(feat_dim, num_classes)

    def _features(self, x4d):
        # x4d: [B, 1, C, T]
        x = self.conv_time(x4d)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        return x

    def forward(self, x):
        # x: [B, C, T] -> [B, 1, C, T]
        x = x.unsqueeze(1)
        x = self._features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)