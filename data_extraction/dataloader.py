import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class EEGTrialDataset(Dataset):
    """
    Torch Dataset yielding individual EEG trials.
    Each item -> (Tensor[channels, time], metadata_dict)
    """
    def __init__(self, index_df: pd.DataFrame, split: str = "train", normalize: bool = True):
        self.df = index_df[index_df["split"] == split].reset_index(drop=True)
        self.normalize = normalize
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _load_npy(self, path: str):
        if path not in self._cache:
            self._cache[path] = np.load(path, mmap_mode="r")
        return self._cache[path]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = self._load_npy(row["eeg_path"])     # shape [n_trials, n_channels, n_timepoints]
        trial_idx = int(row["trial"])
        x = np.array(arr[trial_idx], dtype=np.float32)  # [channels, time]

        if self.normalize:
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

        y = {
            "subject": row["subject"],
            "session": row["session"],
            "run": row["run"],
            "image_name": row["image_name"],
            "class_id": int(row["class_id"]) if not pd.isna(row["class_id"]) else -1,
            "category": row.get("category", None),
        }
        return torch.from_numpy(x), y


def make_dataloaders(index_df: pd.DataFrame, batch_size: int = 32, num_workers: int = 4):
    """
    Create train/val/test DataLoaders.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = EEGTrialDataset(index_df, split=split)
        if len(ds) > 0:
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True
            )
    return loaders