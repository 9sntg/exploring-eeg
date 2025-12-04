# retrieval_metrics.py
import numpy as np
from sklearn.metrics import average_precision_score
from bert_score import score as bert_score
import torch


def recall_at_k_instance(retrieved_idx, gt_idx, k_values=(1,3,5)):
    results = {}
    N = len(gt_idx)

    for k in k_values:
        correct = sum(gt_idx[i] in retrieved_idx[i,:k] for i in range(N))
        results[f"Recall@{k}"] = correct / N
    return results


def recall_at_k_class_aware(retrieved_idx, gt_idx, categories, k_values=(1,3,5)):
    results = {}
    N = len(gt_idx)

    for k in k_values:
        correct = 0
        for i in range(N):
            cat = categories[gt_idx[i]]
            if any(categories[j] == cat for j in retrieved_idx[i,:k]):
                correct += 1
        results[f"Class-aware Recall@{k}"] = correct / N
    return results


def map_caption_level(sim_matrix, gt_idx):
    sim = sim_matrix.cpu().numpy()
    aps = []
    for i, gt in enumerate(gt_idx):
        y = np.zeros(sim.shape[1])
        y[gt] = 1
        aps.append(average_precision_score(y, sim[i]))
    return float(np.mean(aps))


def map_class_aware(sim_matrix, gt_idx, categories):
    sim = sim_matrix.cpu().numpy()
    aps = []

    # category → caption indices
    cat2idx = {}
    for idx, cat in enumerate(categories):
        cat2idx.setdefault(cat, []).append(idx)

    for i, gt in enumerate(gt_idx):
        y = np.zeros(sim.shape[1])
        for j in cat2idx[categories[gt]]:
            y[j] = 1
        aps.append(average_precision_score(y, sim[i]))

    return float(np.mean(aps))


def map_per_class(sim_matrix, gt_idx, categories):
    sim = sim_matrix.cpu().numpy()
    results = {}

    for cat in sorted(set(categories)):
        aps = []
        for i, gt in enumerate(gt_idx):
            if categories[gt] != cat:
                continue
            y = np.zeros(sim.shape[1])
            y[gt] = 1
            aps.append(average_precision_score(y, sim[i]))
        results[cat] = float(np.mean(aps)) if aps else 0.0

    return results


def bertscore_top1(retrieved_captions, gt_captions, sample_idx, device):
    cand = [retrieved_captions[i] for i in sample_idx]
    ref = [gt_captions[i] for i in sample_idx]
    _, _, F1 = bert_score(cand, ref, lang='en', device=str(device), verbose=False)
    return F1.cpu().numpy()