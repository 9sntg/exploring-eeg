# retrieval_visuals.py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def plot_clipscore_distribution(matched_scores, mismatched_scores):
    plt.figure(figsize=(10,6))
    plt.hist(matched_scores, bins=50, alpha=0.6, label="Matched", density=True)
    plt.hist(mismatched_scores, bins=50, alpha=0.6, label="Mismatched", density=True)
    plt.legend()
    plt.xlabel("CLIPScore (cosine similarity)")
    plt.ylabel("Density")
    plt.title("CLIPScore Distribution")
    plt.grid(alpha=0.3)
    plt.show()


def visualize_retrieval_examples(df, retrieved_idx, img_embs, txt_embs, num_examples=5, top_k=3):
    fig, axes = plt.subplots(num_examples, top_k+1, figsize=(16, 3*num_examples))

    N = len(df)
    step = max(1, N // num_examples)

    for i in range(num_examples):
        idx = i * step
        img_path = df.iloc[idx]["image_path"]
        gt_caption = df.iloc[idx]["caption"]

        try:
            img = Image.open(img_path).convert("RGB")
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"GT:\n{gt_caption[:60]}", fontsize=9)
            axes[i, 0].axis("off")
        except:
            axes[i, 0].text(0.5, 0.5, "Image load error", ha="center", va="center")
            axes[i, 0].axis("off")

        for j in range(top_k):
            ridx = retrieved_idx[idx, j]
            caption = df.iloc[ridx]["caption"]
            sim = (img_embs[idx:idx+1] @ txt_embs[ridx:ridx+1].T).item()
            color = "green" if ridx == idx else "red"
            axes[i, j+1].text(
                0.1, 0.5,
                f"Rank {j+1}\nSim={sim:.3f}\n{caption[:90]}",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.2),
                fontsize=9
            )
            axes[i, j+1].axis("off")

    plt.tight_layout()
    plt.show()