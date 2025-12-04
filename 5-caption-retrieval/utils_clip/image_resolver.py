# image_resolver.py
import os
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd


def find_image_path(image_name: str, image_dirs: list[Path]):
    """Find image path across several directories and extensions."""
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    for d in image_dirs:
        for ext in exts:
            p = d / f"{image_name}{ext}"
            if p.exists():
                return str(p)

        # Try raw name
        p = d / image_name
        if p.exists():
            return str(p)

    return None


def build_image_caption_df(captions_df: pd.DataFrame, image_dirs: list[Path]) -> pd.DataFrame:
    """
    Add image_path column to captions_df using find_image_path.
    """
    paths = []
    for img_name in tqdm(captions_df["image_name"], desc="Resolving image paths"):
        p = find_image_path(img_name, image_dirs)
        paths.append(p)

    captions_df["image_path"] = paths
    valid = captions_df[captions_df["image_path"].notna()].reset_index(drop=True)
    return valid