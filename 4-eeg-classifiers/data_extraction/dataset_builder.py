# data_extraction/dataset_builder.py

import os
import re
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd


# 20-category mapping for Pascal/ImageNet-style labels
CLASS_DICT: Dict[str, int] = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'flower': 12, 'horse': 13, 'motorbike': 14,
    'person': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}


def find_runs(root: Path, verbose: bool = True) -> List[Path]:
    """
    Find all EEG .npy run files under the given root.

    Args:
        root: Root directory of the ds005589 dataset.
        verbose: If True, prints how many run files were found.

    Returns:
        Sorted list of Paths to *_1000Hz.npy files.
    """
    run_files = sorted(root.rglob("*_1000Hz.npy"))
    if verbose:
        print(f"Found {len(run_files)} EEG run files under {root}")
    return run_files


def extract_subject_session_run(path: Path) -> Tuple[str, str, str]:
    """
    Extract subject, session, and run identifiers from an EEG filename.

    Example:
        sub-02_ses-01_task-lowSpeed_run-01_1000Hz.npy
        -> ('sub-02', 'ses-01', 'run-01')
    """
    name = path.name
    m_sub = re.search(r"(sub-\d+)", name)
    m_ses = re.search(r"(ses-\d+)", name)
    m_run = re.search(r"(run-\d+)", name)
    sub = m_sub.group(1) if m_sub else "sub-unknown"
    ses = m_ses.group(1) if m_ses else "ses-unknown"
    run = m_run.group(1) if m_run else "run-unknown"
    return sub, ses, run


def canonical_image_key_from_eeg_path(image_path_str: str) -> str:
    """
    Extract canonical image key (to match captions.txt) from EEG image paths.

    Handles Windows-style paths like:
      C:\\Users\\Huawei\\Desktop\\eeg_pascal_imagenet\\pic_10000_resized\\bird\\n02835271_1031_resized.jpg

    Returns:
      'n02835271_1031'
    """
    # Normalize separators
    path_str = str(image_path_str).replace("\\", "/")

    # Base filename
    base = os.path.basename(path_str)

    # Strip extension and '_resized' suffix
    base = re.sub(r"\.(jpg|jpeg|png)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_resized$", "", base, flags=re.IGNORECASE)

    return base.strip()


def load_captions(
    captions_path: Path,
    class_dict: Dict[str, int] = CLASS_DICT,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load captions.txt (tab-separated) and return a cleaned DataFrame with:

      ['image_key', 'category', 'class_id', 'abstracted']
    """
    df = pd.read_csv(captions_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    # Remove extensions from image_name to get key
    df["image_key"] = df["image_name"].astype(str).str.replace(
        r"\.(jpg|jpeg|png)$", "", regex=True
    )

    df["class_id"] = df["category"].str.lower().map(class_dict)

    if verbose:
        print(f"Loaded {len(df)} caption entries from {captions_path.name}")

    return df[["image_key", "category", "class_id", "abstracted"]]


def build_trial_index(
    root: Path,
    captions_path: Path,
    class_dict: Dict[str, int] = CLASS_DICT,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build an index of EEG trials and merge with caption labels.

    Returns a DataFrame with columns:
      ['subject', 'session', 'run', 'trial',
       'eeg_path', 'csv_path',
       'image_name', 'image_key',
       'category', 'class_id', 'abstracted']
    """
    run_files = find_runs(root, verbose=verbose)
    rows = []

    for npy_path in run_files:
        csv_path = Path(str(npy_path).replace("_1000Hz.npy", "_image.csv"))
        if not csv_path.exists():
            continue

        sub, ses, run = extract_subject_session_run(npy_path)
        df_map = pd.read_csv(csv_path)
        df_map.columns = [c.strip() for c in df_map.columns]

        # Find the column with image paths (usually 'FilePath')
        path_col = next(
            (c for c in df_map.columns if "path" in c.lower()),
            df_map.columns[0]
        )

        for trial_idx, row in df_map.iterrows():
            image_path = str(row[path_col])
            image_name = os.path.basename(image_path)
            image_key = canonical_image_key_from_eeg_path(image_path)

            rows.append(
                {
                    "subject": sub,
                    "session": ses,
                    "run": run,
                    "trial": int(trial_idx),
                    "eeg_path": str(npy_path),
                    "csv_path": str(csv_path),
                    "image_name": image_name,
                    "image_key": image_key,
                }
            )

    index_df = pd.DataFrame(rows)
    if verbose:
        print(f"Indexed {len(index_df)} EEG trials from {len(run_files)} runs.")

    captions_df = load_captions(captions_path, class_dict=class_dict, verbose=verbose)
    index_df = index_df.merge(captions_df, on="image_key", how="left")
    if verbose:
        print(f"Merged {len(captions_df)} labels from captions.txt.")

    before = len(index_df)
    # Keep only rows with valid class_id in [0, 19]
    index_df = index_df[index_df["class_id"].between(0, 19)].reset_index(drop=True)
    after = len(index_df)
    if verbose:
        ratio = (after / before) if before > 0 else 0.0
        print(f"Filtered labeled EEG trials: {after}/{before} remain ({ratio:.1%})")

    return index_df


def split_by_session(
    index_df: pd.DataFrame,
    train_sessions: int = 3,
    val_sessions: int = 1,
    test_sessions: int = 1,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Split dataset by session per subject into train/val/test sets.

    A given subject's sessions are randomly permuted, then the first N go to
    train, the next to val, and the next to test.
    """
    rng = np.random.default_rng(seed)
    df = index_df.copy()
    df["split"] = "unassigned"

    for sub, sub_df in df.groupby("subject"):
        sessions = sorted(sub_df["session"].unique())
        rng.shuffle(sessions)

        train_s = set(sessions[:train_sessions])
        val_s = set(sessions[train_sessions:train_sessions + val_sessions])
        test_s = set(
            sessions[train_sessions + val_sessions:
                     train_sessions + val_sessions + test_sessions]
        )

        df.loc[(df["subject"] == sub) & (df["session"].isin(train_s)), "split"] = "train"
        df.loc[(df["subject"] == sub) & (df["session"].isin(val_s)), "split"] = "val"
        df.loc[(df["subject"] == sub) & (df["session"].isin(test_s)), "split"] = "test"

    if verbose:
        print("Split summary:")
        print(df["split"].value_counts())

    return df