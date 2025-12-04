# 4-eeg-classifiers/utils/train_eeg_clip.py

import torch
import torch.nn as nn
from tqdm.auto import tqdm

def cosine_alignment_loss(z_eeg: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """
    L_cos = 1 - cos(eeg, text) averaged over batch.
    Both inputs should already be L2-normalized, but we normalize again for safety.
    """
    z_eeg = torch.nn.functional.normalize(z_eeg, p=2, dim=-1)
    z_txt = torch.nn.functional.normalize(z_txt, p=2, dim=-1)

    cos_sim = (z_eeg * z_txt).sum(dim=-1)  # [B]
    loss = 1.0 - cos_sim.mean()
    return loss


def train_eeg_clip_projection(
    model,
    eeg_loader,
    text_embeddings,      # torch.Tensor [N_text, d]
    subj2idx,             # dict: subject string -> int
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """
    Train EEG->CLIP projection with cosine alignment loss.

    Assumes:
      - eeg_loader yields (x, meta) where:
           x: [B, C, T]
           meta["subject"]: list of subject IDs (strings)
           meta["clip_idx"]: list of indices into text_embeddings
      - text_embeddings: CLIP text embeddings (frozen)
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    text_embeddings = text_embeddings.to(device)
    print("Starting EEG→CLIP projection training...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        seen = 0

        pbar = tqdm(eeg_loader, desc=f"Epoch {epoch}/{epochs}")
        for xb, meta in pbar:
            x = xb.to(device)  # [B, C, T]

            # subject indices
            subj_names = meta["subject"]      # list of strings
            subj_idx = torch.tensor(
                [subj2idx[s] for s in subj_names],
                dtype=torch.long,
                device=device,
            )

            # clip indices for each trial
            clip_idx = torch.tensor(
                meta["clip_idx"],  # list of ints
                dtype=torch.long,
                device=device,
            )

            # get corresponding text embeddings
            z_txt = text_embeddings[clip_idx]   # [B, d_clip]

            # forward EEG -> CLIP
            z_eeg = model(x, subj_idx)          # [B, d_clip]

            loss = cosine_alignment_loss(z_eeg, z_txt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            epoch_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=epoch_loss / max(seen, 1))

        epoch_loss /= max(seen, 1)
        print(f"Epoch {epoch:02d} | Mean alignment loss: {epoch_loss:.4f}")

    print("EEG→CLIP projection training finished.")
    return model