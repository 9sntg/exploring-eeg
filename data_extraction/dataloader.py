import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class EEGTrialDataset(Dataset):
    """
    Yields (x: FloatTensor[C, T], y: dict(..., 'class_id': int)).
    """
    def __init__(self, index_df: pd.DataFrame, split: str = "train", normalize: bool = True):
        df = index_df[index_df["split"] == split].copy()
        # keep only labeled in the dataset itself to avoid empty splits silently
        df = df[df["class_id"].notna()]
        df["class_id"] = df["class_id"].astype(int)
        self.df = df.reset_index(drop=True)
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
        arr = self._load_npy(row["eeg_path"])
        x = np.asarray(arr[int(row["trial"])], dtype=np.float32)  # [C, T]

        if self.normalize:
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

        y = {
            "subject": row["subject"],
            "session": row["session"],
            "run": row["run"],
            "image_name": row["image_name"],
            "class_id": int(row["class_id"]),
        }
        return torch.from_numpy(x), y


def _simple_collate(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, C, T]
    # pack labels into a tensor; keep other fields as lists for reference
    labels = torch.tensor([y["class_id"] for y in ys], dtype=torch.long)
    meta = {k: [y[k] for y in ys] for k in ys[0].keys() if k != "class_id"}
    meta["class_id"] = labels
    return x, meta


def make_dataloaders(index_df: pd.DataFrame, batch_size: int = 128, num_workers: int = 4):
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = EEGTrialDataset(index_df, split=split)
        if len(ds) > 0:
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=(split == "train"),
                num_workers=num_workers, pin_memory=True, collate_fn=_simple_collate
            )
    return loaders