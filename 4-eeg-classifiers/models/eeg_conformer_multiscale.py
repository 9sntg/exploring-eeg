# 4-eeg-classifiers/models/eeg_conformer_multiscale.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Positional Encoding
# -------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)            # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                          # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T]


# -------------------------
# Multi-scale Conv Frontend
# -------------------------

class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale temporal convolution frontend for EEG.

    Input:  [B, C, T]
    Output: [B, T', D_model]

    Branches:
      - Conv1d kernel=7
      - Conv1d kernel=15
      - Conv1d kernel=31
    Concatenated, then:
      - Depthwise separable conv
      - 1x1 projection to transformer_dim
      - Optional temporal downsampling via MaxPool1d
    """

    def __init__(
        self,
        in_channels: int = 122,
        branch_channels: int = 64,
        transformer_dim: int = 256,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Three temporal branches with different kernel sizes
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )

        multiscale_channels = branch_channels * 3

        # Depthwise + pointwise (separable) conv over time
        self.depthwise = nn.Conv1d(
            multiscale_channels,
            multiscale_channels,
            kernel_size=3,
            padding=1,
            groups=multiscale_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm1d(multiscale_channels)
        self.pointwise = nn.Conv1d(
            multiscale_channels,
            transformer_dim,
            kernel_size=1,
            bias=False,
        )

        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)
        self.dropout = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(transformer_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] or [B, T, C]

        Returns:
            seq: [B, T', D_model]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T] or [B, T, C], got {x.shape}")

        # Ensure channels are at dim=1
        # If x is [B, T, C], transpose to [B, C, T]
        if x.shape[1] != 122:
            x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]

        # Multi-scale branches
        b1 = self.branch1(x)   # [B, Bc, T]
        b2 = self.branch2(x)   # [B, Bc, T]
        b3 = self.branch3(x)   # [B, Bc, T]

        h = torch.cat([b1, b2, b3], dim=1)  # [B, 3*Bc, T]

        # Depthwise + pointwise
        h = self.depthwise(h)
        h = self.depthwise_bn(h)
        h = F.gelu(h)
        h = self.pointwise(h)              # [B, D_model, T]

        # Optional temporal pooling
        h = self.pool(h)                   # [B, D_model, T']

        h = h.transpose(1, 2)              # [B, T', D_model]
        h = self.out_ln(h)
        h = self.dropout(h)
        return h


# -------------------------
# Conformer Blocks
# -------------------------

class FeedForwardModule(nn.Module):
    """Positionwise feed-forward as used in Conformer."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    """
    Conformer-style convolution module.

    x: [B, T, D]
    """
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x_ln = self.ln(x)
        x_t = x_ln.transpose(1, 2)  # [B, D, T]

        # First pointwise conv + GLU
        x_pw = self.pointwise_conv1(x_t)   # [B, 2D, T]
        x_glu = F.glu(x_pw, dim=1)         # [B, D, T]

        # Depthwise + BN + Swish
        x_dw = self.depthwise_conv(x_glu)  # [B, D, T]
        x_dw = self.bn(x_dw)
        x_dw = x_dw * torch.sigmoid(x_dw)  # Swish

        # Final pointwise conv
        x_out = self.pointwise_conv2(x_dw)  # [B, D, T]
        x_out = x_out.transpose(1, 2)       # [B, T, D]
        return self.dropout(x_out)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x_norm = self.ln(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False)
        return self.dropout(attn_out)


class ConformerBlock(nn.Module):
    """
    Full Conformer block:
      x = x + 0.5 * FF(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 0.5 * FF(x)
      x = LN(x)
    """
    def __init__(
        self,
        dim: int,
        ff_mult: int = 4,
        n_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, mult=ff_mult, dropout=dropout)
        self.attn = MultiHeadSelfAttentionModule(dim, n_heads=n_heads, dropout=dropout)
        self.conv = ConvModule(dim, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, mult=ff_mult, dropout=dropout)
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FF + residual (1/2)
        x = x + 0.5 * self.ff1(x)
        # MHSA + residual
        x = x + self.attn(x)
        # Conv + residual
        x = x + self.conv(x)
        # FF + residual (1/2)
        x = x + 0.5 * self.ff2(x)
        # Final LN
        x = self.final_ln(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        ff_mult: int = 4,
        n_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=dim,
                    ff_mult=ff_mult,
                    n_heads=n_heads,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]

        Returns:
            h: [B, T, D]
        """
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------
# Top-level Model
# -------------------------

class ConformerMultiScaleEEG(nn.Module):
    """
    Conformer-based multi-scale EEG model with per-subject heads.

    Stages:
      1. MultiScaleConvBlock (temporal multi-kernel + depthwise separable)
      2. ConformerEncoder (MHSA + conv modules)
      3. Global pooling
      4. Subject-specific classifier heads

    Methods:
      - forward_features(x): [B, C, T] -> [B, D]
      - forward(x, subj_idx): -> [B, num_classes]
    """

    def __init__(
        self,
        num_channels: int = 122,
        num_classes: int = 20,
        n_subjects: int = 13,
        hidden_dim: int = 192,          # branch_channels * 3 → then projected to transformer_dim
        transformer_dim: int = 256,
        n_layers: int = 8,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.n_subjects = n_subjects
        self.transformer_dim = transformer_dim

        # Multi-scale frontend
        # We choose branch_channels such that 3 * branch_channels is roughly hidden_dim
        branch_channels = hidden_dim // 3
        branch_channels = max(branch_channels, 32)

        self.frontend = MultiScaleConvBlock(
            in_channels=num_channels,
            branch_channels=branch_channels,
            transformer_dim=transformer_dim,
            pool_kernel=2,
            pool_stride=2,      # 500 → 250
            dropout=dropout,
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(
            dim=transformer_dim,
            n_layers=n_layers,
            ff_mult=ff_mult,
            n_heads=n_heads,
            conv_kernel=31,
            dropout=dropout,
            max_len=300,        # safe for 250 time steps
        )

        # Subject-specific classifier heads
        self.subject_heads = nn.ModuleList(
            [nn.Linear(transformer_dim, num_classes) for _ in range(n_subjects)]
        )

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    # --------- Shared embedding used for CLIP alignment / export ---------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] or [B, T, C]

        Returns:
            z: [B, transformer_dim] global EEG embedding
        """
        # 1) Multi-scale temporal conv frontend
        h = self.frontend(x)            # [B, T', D]

        # 2) Conformer encoder
        h = self.encoder(h)             # [B, T', D]

        # 3) Global average pooling over time
        z = h.mean(dim=1)               # [B, D]

        return z

    # Alias for backward compatibility if something calls model.encode(...)
    def encode(self, x: torch.Tensor, subj_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_features(x)

    # --------- Classification forward (per-subject heads) ---------

    def forward(self, x: torch.Tensor, subj_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        [B, C, T] or [B, T, C]
            subj_idx: [B] LongTensor with values in [0, n_subjects-1]

        Returns:
            logits:   [B, num_classes]
        """
        if subj_idx.dim() != 1 or subj_idx.size(0) != x.size(0):
            raise ValueError(f"Expected subj_idx shape [B], got {subj_idx.shape}")

        feats = self.forward_features(x)             # [B, D]
        B = feats.size(0)
        device = feats.device

        logits = feats.new_zeros((B, self.num_classes))  # [B, num_classes]

        # Route each sample to correct subject head
        for s in subj_idx.unique():
            mask = (subj_idx == s)
            if mask.any():
                head = self.subject_heads[int(s)]
                logits[mask] = head(feats[mask])

        return logits