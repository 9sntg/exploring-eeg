# models/eeg_clip_projection.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGToCLIPProjection(nn.Module):
    """
    Wraps a trained EEG encoder (e.g., MultiHeadEEGClassifier)
    and adds a projection from EEG hidden_dim -> CLIP embedding dim.

    By default:
      - CLIP is frozen (we only need text embeddings)
      - EEG encoder can be frozen or fine-tuned (configurable)
    """
    def __init__(
        self,
        eeg_encoder: nn.Module,
        hidden_dim: int,
        clip_dim: int = 512,
        freeze_eeg: bool = True,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.hidden_dim = hidden_dim
        self.clip_dim = clip_dim
        self.freeze_eeg = freeze_eeg

        # Simple 2-layer MLP projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim),
        )

        # Optionally freeze EEG encoder params
        if freeze_eeg:
            for p in self.eeg_encoder.parameters():
                p.requires_grad = False

    def encode_eeg(self, x: torch.Tensor, subj_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized EEG embedding in CLIP space.
        """
        if self.freeze_eeg:
            with torch.no_grad():
                h = self.eeg_encoder.forward_features(x)   # [B, H]
        else:
            h = self.eeg_encoder.forward_features(x)

        z = self.proj(h)                   # [B, clip_dim]
        z = F.normalize(z, p=2, dim=-1)    # unit norm
        return z

    def forward(self, x: torch.Tensor, subj_idx: torch.Tensor) -> torch.Tensor:
        """
        Alias for encode_eeg, so you can call model(x, subj_idx).
        """
        return self.encode_eeg(x, subj_idx)