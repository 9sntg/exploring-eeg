import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# ✅ Fixed dataset root
DATASET_ROOT = Path("/ocean/projects/cis250019p/gandotra/11785-gp-eeg/ds005589")
CAPTIONS_PATH = Path("/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt")

# 20-class mapping
CLASS_DICT = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'flower': 12, 'horse': 13, 'motorbike': 14,
    'person': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------

def find_runs(root: Path) -> list[Path]:
    """Find all *_1000Hz.npy EEG run files."""
    return sorted(root.rglob("*_1000Hz.npy"))


def extract_subject_session_run(path: Path) -> Tuple[str, str, str]:
    """Extract subject, session, and run identifiers from file name."""
    name = path.name
    m_sub = re.search(r"(sub-\d+)", name)
    m_ses = re.search(r"(ses-\d+)", name)
    m_run = re.search(r"(run-\d+)", name)
    sub = m_sub.group(1) if m_sub else "sub-unknown"
    ses = m_ses.group(1) if m_ses else "ses-unknown"
    run = m_run.group(1) if m_run else "run-unknown"
    return sub, ses, run


def load_captions(captions_path: Path = CAPTIONS_PATH) -> pd.DataFrame:
    """
    Load captions.txt (tab-separated) and return cleaned DataFrame:
    columns = ['image_key', 'category', 'class_id', 'abstracted']
    """
    df = pd.read_csv(captions_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    # Remove file extensions from image_name to get a join key
    df["image_key"] = df["image_name"].astype(str).str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)

    # Map category -> numeric class_id
    df["class_id"] = df["category"].str.lower().map(CLASS_DICT)

    print(f"✅ Loaded {len(df)} caption entries from {captions_path.name}")
    return df[["image_key", "category", "class_id", "abstracted"]]
    

# --------------------------------------------------------------------
# Main indexing function
# --------------------------------------------------------------------

def build_trial_index(
    root: Path = DATASET_ROOT,
    captions_path: Path = CAPTIONS_PATH
) -> pd.DataFrame:
    """
    Build a trial-level index for EEG data and merge with category labels.
    Returns a DataFrame with columns:
    ['subject', 'session', 'run', 'trial', 'eeg_path', 'image_name', 'image_key', 'category', 'class_id', 'abstracted']
    """
    run_files = find_runs(root)
    if not run_files:
        raise FileNotFoundError(f"No EEG .npy files found under {root}")

    rows = []
    for npy_path in run_files:
        csv_path = Path(str(npy_path).replace("_1000Hz.npy", "_image.csv"))
        if not csv_path.exists():
            continue

        sub, ses, run = extract_subject_session_run(npy_path)
        df_map = pd.read_csv(csv_path)
        df_map.columns = [c.strip() for c in df_map.columns]

        # Identify image path column
        path_col = next((c for c in df_map.columns if "path" in c.lower()), df_map.columns[0])

        for trial_idx, row in df_map.iterrows():
            image_name = os.path.basename(str(row[path_col]))
            image_key = os.path.splitext(image_name)[0]
            rows.append({
                "subject": sub,
                "session": ses,
                "run": run,
                "trial": int(trial_idx),
                "eeg_path": str(npy_path),
                "csv_path": str(csv_path),
                "image_name": image_name,
                "image_key": image_key
            })

    index_df = pd.DataFrame(rows)
    print(f"✅ Indexed {len(index_df)} EEG trials from {len(run_files)} runs.")

    # Merge with captions
    if captions_path.exists():
        captions_df = load_captions(captions_path)
        index_df = index_df.merge(captions_df, on="image_key", how="left")
        print(f"✅ Merged {len(captions_df)} labels from captions.txt with EEG trials.")
    else:
        print(f"⚠️ captions.txt not found at {captions_path}")

    return index_df


# --------------------------------------------------------------------
# Split function
# --------------------------------------------------------------------

def split_by_session(index_df: pd.DataFrame,
                     train_sessions: int = 3,
                     val_sessions: int = 1,
                     test_sessions: int = 1,
                     seed: int = 42) -> pd.DataFrame:
    """
    Split dataset by session within each subject.
    """
    rng = np.random.default_rng(seed)
    df = index_df.copy()
    df["split"] = "unassigned"

    for sub, sub_df in df.groupby("subject"):
        sessions = sorted(sub_df["session"].unique())
        rng.shuffle(sessions)
        train_s = set(sessions[:train_sessions])
        val_s   = set(sessions[train_sessions:train_sessions + val_sessions])
        test_s  = set(sessions[train_sessions + val_sessions:
                               train_sessions + val_sessions + test_sessions])

        df.loc[(df["subject"] == sub) & (df["session"].isin(train_s)), "split"] = "train"
        df.loc[(df["subject"] == sub) & (df["session"].isin(val_s)), "split"]   = "val"
        df.loc[(df["subject"] == sub) & (df["session"].isin(test_s)), "split"]  = "test"

    print(df["split"].value_counts())
    return df