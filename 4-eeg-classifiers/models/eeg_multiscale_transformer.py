# /jet/home/bermudez/exploring-eeg/4-eeg-classifiers/models/eeg_multiscale_transformer_mh.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
#  Positional Encoding (same as your previous implementation)
# ------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ------------------------------------------------------------
#  Multi-scale temporal convolution block
# ------------------------------------------------------------
class MultiScaleConvBlock(nn.Module):
    """
    Produces feature maps with three temporal kernels: 7, 15, 31.
    Output dims = hidden_dim (per branch), concatenated.
    """
    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()

        self.branch7 = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.branch15 = nn.Conv1d(in_channels, hidden_dim, kernel_size=15, padding=7)
        self.branch31 = nn.Conv1d(in_channels, hidden_dim, kernel_size=31, padding=15)

        self.bn = nn.BatchNorm1d(hidden_dim * 3)
        self.act = nn.GELU()

    def forward(self, x):
        b7 = self.branch7(x)
        b15 = self.branch15(x)
        b31 = self.branch31(x)

        x = torch.cat([b7, b15, b31], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x  # [B, 3*hidden_dim, T]


# ------------------------------------------------------------
#  Depthwise + Separable Conv block (EEGNet-style)
# ------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ------------------------------------------------------------
#  Transformer Backbone
# ------------------------------------------------------------
class TransformerBackbone(nn.Module):
    def __init__(
        self, hidden_dim: int, n_layers: int, n_heads: int,
        ff_mult: int, dropout: float
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.pos = PositionalEncoding(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pos(x)
        x = self.drop(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return x  # [B, H]


# ------------------------------------------------------------
#  Full Multi-head Multi-scale EEG model
# ------------------------------------------------------------
class MultiScaleEEGTransformerMH(nn.Module):
    """
    Improved model:
      - Multi-scale temporal conv
      - Depthwise + separable conv
      - Transformer backbone
      - Subject-specific classification heads
      - forward_features() for CLIP alignment
    """
    def __init__(
        self,
        num_channels: int = 122,
        num_classes: int = 20,
        n_subjects: int = 13,
        hidden_dim: int = 128,  # per branch; final conv_dim = 3*hidden_dim
        transformer_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        conv_out_dim = hidden_dim * 3

        # Stage 1 — Multi-scale temporal filtering
        self.msconv = MultiScaleConvBlock(
            in_channels=num_channels,
            hidden_dim=hidden_dim
        )

        # Stage 2 — Depthwise + Separable
        self.dws = DepthwiseSeparableConv(conv_out_dim)

        # Reduce conv output to transformer_dim
        self.proj = nn.Linear(conv_out_dim, transformer_dim)

        # Stage 3 — Transformer backbone
        self.backbone = TransformerBackbone(
            hidden_dim=transformer_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout
        )

        # Stage 4 — subject-specific classifier heads
        self.subject_heads = nn.ModuleList(
            [nn.Linear(transformer_dim, num_classes) for _ in range(n_subjects)]
        )

        self.transformer_dim = transformer_dim

    # --------- Shared embedding for CLIP alignment ---------
    def forward_features(self, x):
        """
        x: [B, C, T]
        returns: [B, transformer_dim]
        """
        if x.shape[1] != 122:
            x = x.transpose(1, 2)

        x = self.msconv(x)            # [B, 3*hidden_dim, T]
        x = self.dws(x)               # [B, 3*hidden_dim, T]
        x = x.transpose(1, 2)         # [B, T, 3*hidden_dim]

        x = self.proj(x)              # [B, T, transformer_dim]
        z = self.backbone(x)          # [B, transformer_dim]

        return z

    # --------- Classification (uses subject-specific heads) ---------
    def forward(self, x, subj_idx):
        feats = self.forward_features(x)
        B = feats.size(0)

        logits = feats.new_zeros(B, self.subject_heads[0].out_features)

        for s in subj_idx.unique():
            mask = (subj_idx == s)
            logits[mask] = self.subject_heads[int(s)](feats[mask])

        return logits