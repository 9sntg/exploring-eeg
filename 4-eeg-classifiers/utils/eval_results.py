# eval_results.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# 20-class mapping used across the EEG dataset
CLASS_NAMES = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','flower','horse','motorbike',
    'person','sheep','sofa','train','tvmonitor'
]


# -------------------------------------------------------------------------
# 1. COLLECT MODEL PREDICTIONS
# -------------------------------------------------------------------------
def collect_predictions(model, dataloader, device="cpu"):
    """
    Runs inference on a dataloader and returns:
        preds: numpy array of predicted class indices
        labels: numpy array of true class indices
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for xb, meta in dataloader:
            x = xb.to(device)
            y = meta["class_id"].to(device)
            logits = model(x)
            p = logits.argmax(dim=1)

            preds.append(p.cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    return preds, labels


# -------------------------------------------------------------------------
# 2. PLOT CONFUSION MATRIX
# -------------------------------------------------------------------------
def plot_confusion_matrix(labels, preds, normalize=True, save_path=None):
    cm = confusion_matrix(labels, preds, labels=range(len(CLASS_NAMES)))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        cmap="viridis",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Normalized frequency'}
    )
    plt.title("Confusion Matrix — EEG Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved at {save_path}")

    plt.show()

    return cm


# -------------------------------------------------------------------------
# 3. PER-CLASS ACCURACY TABLE
# -------------------------------------------------------------------------
def compute_per_class_accuracy(labels, preds):
    cm = confusion_matrix(labels, preds, labels=range(len(CLASS_NAMES)))
    cm_sum = cm.sum(axis=1, keepdims=True)
    class_acc = np.diag(cm) / cm_sum.squeeze()

    df = pd.DataFrame({
        "Category": CLASS_NAMES,
        "Accuracy (%)": (class_acc * 100).round(2)
    }).sort_values("Accuracy (%)", ascending=False)

    return df


def save_latex_table(df, file_path):
    """Save DataFrame as LaTeX table."""
    with open(file_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    print(f"LaTeX table saved to {file_path}")


# -------------------------------------------------------------------------
# 4. GLOBAL METRICS (accuracy, precision, recall, F1)
# -------------------------------------------------------------------------
def compute_global_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    metrics = {
        "Accuracy (%)": acc * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1-score (%)": f1 * 100
    }
    return metrics


def metrics_to_dataframe(metrics: dict):
    """Convert dict of scalar metrics to a table."""
    df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value (%)": list(metrics.values())
    })
    return df


# -------------------------------------------------------------------------
# 5. TOP CONFUSION PAIRS
# -------------------------------------------------------------------------
def top_confusion_pairs(labels, preds, top_k=10):
    cm = confusion_matrix(labels, preds, labels=range(len(CLASS_NAMES)))
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)

    pairs = []
    for i in range(20):
        for j in range(20):
            if i != j:
                pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], off_diag[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs_sorted[:top_k]


# -------------------------------------------------------------------------
# 6. FULL EVALUATION PIPELINE (one call)
# -------------------------------------------------------------------------
def evaluate_model(model, dataloader, device="cpu", save_dir=None):
    """
    Runs full evaluation:
        - Collect predictions
        - Confusion matrix
        - Per-class accuracy table
        - Global metrics
        - Top confusion errors
    Returns all objects (cm, tables) for further use.
    """

    print("\n Collecting predictions...")
    preds, labels = collect_predictions(model, dataloader, device)
    print(f"Collected {len(preds)} predictions.")

    print("\n Generating confusion matrix...")
    cm = plot_confusion_matrix(labels, preds, normalize=True)

    print("\n Computing per-class accuracy...")
    per_class_df = compute_per_class_accuracy(labels, preds)
    print(per_class_df)

    print("\n Computing global metrics...")
    global_metrics = compute_global_metrics(labels, preds)
    summary_df = metrics_to_dataframe(global_metrics)
    print(summary_df)

    print("\n Identifying most common confusion pairs...")
    confusion_pairs = top_confusion_pairs(labels, preds)
    confusion_df = pd.DataFrame(confusion_pairs, columns=["True", "Predicted", "Count"])
    print(confusion_df)

    # Optional saving
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_latex_table(per_class_df, os.path.join(save_dir, "per_class.tex"))
        save_latex_table(summary_df, os.path.join(save_dir, "summary.tex"))
        confusion_df.to_csv(os.path.join(save_dir, "top_confusions.csv"), index=False)

    return cm, per_class_df, summary_df, confusion_df