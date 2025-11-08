# dataset_builder.py
import os, re
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

# Paths
DATASET_ROOT = Path("/ocean/projects/cis250019p/gandotra/11785-gp-eeg/ds005589")
CAPTIONS_PATH = Path("/ocean/projects/cis250019p/gandotra/11785-gp-eeg/captions.txt")

# 20 Pascal/Imagenet categories used in this paradigm
CLASS_DICT = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'flower': 12, 'horse': 13, 'motorbike': 14,
    'person': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

def find_runs(root: Path) -> list[Path]:
    return sorted(root.rglob("*_1000Hz.npy"))

def extract_subject_session_run(path: Path) -> Tuple[str, str, str]:
    name = path.name
    sub = re.search(r"(sub-\d+)", name)
    ses = re.search(r"(ses-\d+)", name)
    run = re.search(r"(run-\d+)", name)
    return (sub.group(1) if sub else "sub-unknown",
            ses.group(1) if ses else "ses-unknown",
            run.group(1) if run else "run-unknown")

def canonical_image_key_from_eeg_path(image_path_str: str) -> str:
    """
    From EEG image path like:
      C:\\...\\pic_10000_resized\\bird\\004461_resized.jpg
      .../pic_10000_resized/train/2009_000920_resized.jpg
    -> '004461' or '2009_000920'
    """
    base = os.path.basename(str(image_path_str)).replace("\\", "/")
    base = re.sub(r"\.(jpg|jpeg|png)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_resized$", "", base)  # strip the trailing _resized
    return base

def load_captions(captions_path: Path = CAPTIONS_PATH) -> pd.DataFrame:
    """
    captions.txt columns: dataset, category, image_name, abstracted
    We create: image_key (no extension), class_id
    """
    df = pd.read_csv(captions_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    # e.g. 'n02835271_1031.jpg' -> 'n02835271_1031'
    df["image_key"] = df["image_name"].astype(str).str.replace(
        r"\.(jpg|jpeg|png)$", "", regex=True
    )
    df["class_id"] = df["category"].str.lower().map(CLASS_DICT)
    return df[["image_key", "category", "class_id", "abstracted"]]

def build_trial_index(
    root: Path = DATASET_ROOT,
    captions_path: Path = CAPTIONS_PATH
) -> pd.DataFrame:
    """
    Build a trial-level index for EEG data and merge with category labels.
    Filters unlabeled EEGs automatically after merge.
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
        path_col = next((c for c in df_map.columns if "path" in c.lower()), df_map.columns[0])

        for trial_idx, row in df_map.iterrows():
            image_path = str(row[path_col])
            image_name = os.path.basename(image_path)
            image_key = canonical_image_key_from_eeg_path(image_path)
            rows.append({
                "subject": sub,
                "session": ses,
                "run": run,
                "trial": int(trial_idx),
                "eeg_path": str(npy_path),
                "csv_path": str(csv_path),
                "image_path": image_path,
                "image_name": image_name,
                "image_key": image_key
            })

    index_df = pd.DataFrame(rows)
    print(f"✅ Indexed {len(index_df)} EEG trials from {len(run_files)} runs.")

    # --- Merge labels ---
    if captions_path.exists():
        caps = load_captions(captions_path)
        index_df = index_df.merge(caps, on="image_key", how="left")
        print(f"✅ Merged {len(caps)} labels from captions.txt.")
    else:
        print(f"⚠️ captions.txt not found at {captions_path}")

    # --- Filter only labeled EEGs ---
    before = len(index_df)
    index_df = index_df[index_df["class_id"].between(0, 19)].reset_index(drop=True)
    after = len(index_df)
    print(f"✅ Filtered labeled EEG trials: {after}/{before} remain ({after/before:.1%})")

    return index_df


def split_by_session(index_df: pd.DataFrame,
                     train_sessions: int = 3, val_sessions: int = 1, test_sessions: int = 1,
                     seed: int = 42) -> pd.DataFrame:
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

# add this tiny helper at the bottom of dataset_builder.py
def keep_labeled_rows(index_df):
    """Keep only rows with valid class_id (0..19)."""
    df = index_df.copy()
    df = df[df["class_id"].notna()].copy()
    df["class_id"] = df["class_id"].astype(int)
    return df