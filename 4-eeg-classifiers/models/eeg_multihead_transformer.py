# /jet/home/bermudez/exploring-eeg/4-eeg-classifiers/models/eeg_multihead_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)            # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                          # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class ConvFeatureExtractor(nn.Module):
    """
    Stage 1: Convolutional block over time.
    Input:  [B, C=122, T=500]
    Output: [B, T', D] where T' < T and D=hidden_dim
    """
    def __init__(self, num_channels: int = 122, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),   # 500 -> 250
        )

    def forward(self, x):
        """
        x: [B, C, T]
        returns: [B, T', D]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B,C,T], got {x.shape}")
        # ensure channels are dim=1
        if x.shape[1] != 122:             # if [B,T,C]
            x = x.transpose(1, 2)
        x = self.conv(x)                  # [B, D, T']
        x = x.transpose(1, 2)             # [B, T', D]
        return x


class SharedBackbone(nn.Module):
    """
    Stage 2: Shared Transformer encoder backbone.
    Input:  [B, T', D]
    Output: [B, D] pooled embedding
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T', D]
        returns: [B, D] global representation
        """
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)           # [B, T', D]
        x = x.mean(dim=1)                 # global average over time
        return x                          # [B, D]


class MultiHeadEEGClassifier(nn.Module):
    """
    Full model:
      - ConvFeatureExtractor (per-trial temporal features)
      - SharedBackbone Transformer (shared across all subjects)
      - Subject-specific linear heads (one per subject)

    forward(x, subj_idx) -> logits [B, num_classes]
    forward_features(x)  -> shared EEG embedding [B, hidden_dim]
    """
    def __init__(
        self,
        num_channels: int = 122,
        num_classes: int = 20,
        n_subjects: int = 13,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.n_subjects = n_subjects
        self.hidden_dim = hidden_dim

        # Stage 1 + 2
        self.feature_extractor = ConvFeatureExtractor(num_channels, hidden_dim)
        self.backbone = SharedBackbone(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_len=300,   # 250 after pooling is safe
        )

        # Stage 3: subject-specific heads
        self.subject_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(n_subjects)]
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    # ---- shared embedding for later CLIP alignment ----
    def forward_features(self, x):
        """
        x: [B, C, T] or [B, T, C]
        returns: [B, hidden_dim]
        """
        x = self.feature_extractor(x)  # [B, T', D]
        x = self.backbone(x)           # [B, D]
        return x

    def forward(self, x, subj_idx):
        """
        x:        [B, C, T] or [B, T, C]
        subj_idx: [B] LongTensor with values in [0, n_subjects-1]
        returns:  logits [B, num_classes]
        """
        if subj_idx.dim() != 1 or subj_idx.size(0) != x.size(0):
            raise ValueError(f"Expected subj_idx shape [B], got {subj_idx.shape}")

        feats = self.forward_features(x)             # [B, D]
        B = feats.size(0)
        device = feats.device

        logits = feats.new_zeros((B, self.num_classes))  # [B, num_classes]

        # apply correct head per subject (keeps gradients clean)
        for s in subj_idx.unique():
            mask = (subj_idx == s)
            if mask.any():
                head = self.subject_heads[int(s)]
                logits[mask] = head(feats[mask])

        return logits