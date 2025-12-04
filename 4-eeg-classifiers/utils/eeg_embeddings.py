# utils/eeg_embeddings.py

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def extract_embeddings_multihead(model, loader, device, subj2idx):
    """
    Extract EEG embeddings using model.forward_features().
    Returns:
      embeddings: [N, hidden_dim]
      meta_df: metadata per trial
    """
    model.eval()
    all_embeds = []
    meta_records = []

    with torch.no_grad():
        for x, meta in loader:
            x = x.to(device)

            # use shared EEG embedding
            z = model.forward_features(x)    # [B, hidden_dim]
            all_embeds.append(z.cpu())

            # store metadata
            B = x.size(0)
            for i in range(B):
                meta_records.append({
                    "subject": meta["subject"][i],
                    "session": meta["session"][i],
                    "run": meta["run"][i],
                    "image_name": meta["image_name"][i],
                    "category": meta["category"][i],
                    "class_id": int(meta["class_id"][i]),
                })

    embeddings = torch.cat(all_embeds, dim=0).numpy()
    meta_df = pd.DataFrame(meta_records)

    print(f"Collected {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return embeddings, meta_df

def export_all_splits_multihead(
    model: torch.nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    subj2idx: Dict[str, int],
    save_dir: str | Path = "eeg_embeddings",
    splits: Tuple[str, ...] = ("train", "val", "test"),
    prefix: str = "multihead",
) -> None:
    """
    Export EEG embeddings + metadata for multiple splits.

    Args:
        model: Trained MultiHeadEEGClassifier (with encode method)
        loaders: dict mapping {"train": dl, "val": dl, "test": dl}
        device: torch.device
        subj2idx: mapping from subject string to int index
        save_dir: base directory where .npy and .csv files are saved
        splits: which splits from loaders to export
        prefix: filename prefix, e.g., "multihead"

    Saves:
        {save_dir}/{prefix}_{split}_embeddings.npy
        {save_dir}/{prefix}_{split}_meta.csv
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in loaders:
            print(f"[export_all_splits_multihead] Split '{split}' not found in loaders, skipping.")
            continue

        print(f"\n=== Exporting split: {split} ===")
        embeddings, meta_df = extract_embeddings_multihead(
            model=model,
            loader=loaders[split],
            device=device,
            subj2idx=subj2idx,
        )

        emb_path = save_dir / f"{prefix}_{split}_embeddings.npy"
        meta_path = save_dir / f"{prefix}_{split}_meta.csv"

        np.save(emb_path, embeddings)
        meta_df.to_csv(meta_path, index=False)

        print(f"Saved embeddings to: {emb_path}")
        print(f"Saved metadata  to: {meta_path}")
        print(f"Shape: embeddings={embeddings.shape}, rows={len(meta_df)}")