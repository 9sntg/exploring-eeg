# caption_loader.py
import pandas as pd
from pathlib import Path

# Reuse EEG caption logic
import sys
sys.path.append('/jet/home/bermudez/exploring-eeg/4-eeg-classifiers')

from data_extraction.dataset_builder import load_captions as load_eeg_captions


def load_captions_for_clip(captions_path: Path) -> pd.DataFrame:
    """
    Load captions using the EEG project function, and adapt for CLIP pipeline.
    Returns a DataFrame with columns:
        - image_name (str)
        - caption (str)
        - category (str)
    """
    df = load_eeg_captions(captions_path)

    # EEG function returns: image_key, category, class_id, abstracted
    df = df.rename(
        columns={
            "image_key": "image_name",
            "abstracted": "caption",
        }
    )
    df = df[["image_name", "caption", "category"]].copy()

    return df