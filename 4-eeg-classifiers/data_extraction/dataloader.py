# data_extraction/dataloader.py

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EEGTrialDataset(Dataset):
    """
    PyTorch Dataset for single-trial EEG data.

    Each item returns:
      x: Tensor [C, T]  (normalized per channel)
      y: dict with metadata including 'class_id'
    """

    def __init__(self, index_df: pd.DataFrame, split: str = "train", normalize: bool = True):
        self.df = index_df[index_df["split"] == split].reset_index(drop=True)
        self.normalize = normalize
        self._cache: Dict[str, np.ndarray] = {}

        if len(self.df) == 0:
            print(f"Warning: {split} split has no samples!")

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, path: str) -> np.ndarray:
        """
        Lazy-load and cache large .npy EEG arrays.

        Expected shape in file: [n_trials, C, T]
        """
        if path not in self._cache:
            self._cache[path] = np.load(path, mmap_mode="r")
        return self._cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        arr = self._load_npy(row["eeg_path"])     # [n_trials, C, T]
        x = np.array(arr[int(row["trial"])], dtype=np.float32)  # [C, T]

        if self.normalize:
            x = (x - x.mean(axis=1, keepdims=True)) / (
                x.std(axis=1, keepdims=True) + 1e-6
            )

        y: Dict[str, Any] = {
            "subject": row["subject"],
            "session": row["session"],
            "run": row["run"],
            "image_name": row["image_name"],
            "category": row["category"],
            "class_id": int(row["class_id"]),
        }

        return torch.from_numpy(x), y


def eeg_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Custom collate function to batch EEG trials and metadata.

    Returns:
      x_batch: Tensor [B, C, T]
      meta: dict with
        - class_id: Tensor [B]
        - other keys: lists of length B
    """
    xs, ys = zip(*batch)
    x_batch = torch.stack(xs, dim=0)  # [B, C, T]
    labels = torch.tensor([y["class_id"] for y in ys], dtype=torch.long)

    meta: Dict[str, Any] = {
        k: [y[k] for y in ys] for k in ys[0].keys() if k != "class_id"
    }
    meta["class_id"] = labels
    return x_batch, meta


def make_dataloaders(
    index_df: pd.DataFrame,
    batch_size: int = 128,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create train, val, and test DataLoaders from the index DataFrame.

    Args:
        index_df: DataFrame produced by build_trial_index + split_by_session.
        batch_size: Batch size for all splits.
        num_workers: Number of workers for DataLoader.

    Returns:
        dict with keys 'train', 'val', 'test' (if they have data).
    """
    loaders: Dict[str, DataLoader] = {}

    for split in ["train", "val", "test"]:
        ds = EEGTrialDataset(index_df, split=split)
        if len(ds) > 0:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=eeg_collate_fn,
            )
            print(f"{split}: {len(ds)} samples")
        else:
            print(f"No samples in {split} split!")

    return loaders