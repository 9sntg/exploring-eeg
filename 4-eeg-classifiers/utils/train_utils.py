# utils/train_utils.py

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn
from tqdm.auto import tqdm


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy for a batch.
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_model(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    epochs: int = 50,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
    max_grad_norm: float = 2.0,
) -> Dict[str, Any]:
    """
    Generic training loop with:
      - train/val per epoch
      - gradient clipping
      - best-model checkpointing by val accuracy

    Returns a history dict with:
      {
        "train_loss": [...],
        "val_loss":   [...],
        "train_acc":  [...],
        "val_acc":    [...],
        "best_epoch": int,
        "best_val_acc": float,
        "best_state_dict": state_dict
      }
    """
    if optimizer is None:
        raise ValueError("optimizer must be provided to train_model().")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_epoch": None,
        "best_val_acc": 0.0,
        "best_state_dict": None,
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        # ----------------- TRAIN -----------------
        model.train()
        train_loss, train_acc, seen = 0.0, 0.0, 0

        if "train" not in loaders:
            raise KeyError("Expected 'train' key in loaders dict.")

        for xb, meta in tqdm(loaders["train"], desc=f"Epoch {epoch}/{epochs}"):
            x = xb.to(device, non_blocking=True)
            y = meta["class_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_acc  += compute_accuracy(logits, y) * bs
            seen       += bs

        train_loss /= max(seen, 1)
        train_acc  /= max(seen, 1)

        # ----------------- VALIDATION -----------------
        model.eval()
        val_loss, val_acc, val_seen = 0.0, 0.0, 0

        if "val" in loaders and len(loaders["val"]) > 0:
            with torch.no_grad():
                for xb, meta in loaders["val"]:
                    x = xb.to(device, non_blocking=True)
                    y = meta["class_id"].to(device, non_blocking=True)
                    logits = model(x)
                    loss = criterion(logits, y)

                    bs = x.size(0)
                    val_loss += loss.item() * bs
                    val_acc  += compute_accuracy(logits, y) * bs
                    val_seen += bs

            val_loss /= max(val_seen, 1)
            val_acc  /= max(val_seen, 1)
        else:
            # no val loader -> just mirror train stats
            val_loss = train_loss
            val_acc  = train_acc

        # step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        # log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            if checkpoint_dir is not None:
                ckpt_path = checkpoint_dir / f"{model.__class__.__name__}_best_epoch{epoch:02d}.pth"
                torch.save(best_state, ckpt_path)
                print(f"Saved new best model at {ckpt_path} (val acc={val_acc*100:.2f}%)")

        # epoch summary
        print(
            f"Epoch {epoch:02d}: "
            f"Train loss={train_loss:.4f} | acc={train_acc*100:.2f}% | "
            f"Val loss={val_loss:.4f} | acc={val_acc*100:.2f}%"
        )

    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc
    history["best_state_dict"] = best_state

    print(
        f"\Training complete. Best epoch {best_epoch} "
        f"with val acc={best_val_acc*100:.2f}%"
    )

    return history

def train_multihead_model(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    checkpoint_dir: Path | str = "checkpoints_multihead",
    max_grad_norm: float | None = 2.0,
    subj2idx: dict | None = None,
):
    """
    Training loop for multi-head EEG model:
      model(x, subj_idx) -> logits

    Args:
        model:        MultiHeadEEGClassifier (or compatible)
        loaders:      dict with "train" and "val" DataLoaders
        device:       torch.device
        epochs:       number of epochs
        optimizer:    optimizer
        criterion:    loss function (e.g., CrossEntropyLoss)
        scheduler:    optional LR scheduler
        checkpoint_dir: directory to save best checkpoint
        max_grad_norm: gradient clipping value (None to disable)
        subj2idx:     dict mapping subject string -> integer head index

    Returns:
        history: dict with training/validation curves and best model info
    """
    if subj2idx is None:
        raise ValueError("train_multihead_model requires a subj2idx mapping.")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": None,
        "best_val_acc": None,
        "best_state_dict": None,
    }

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ----------- TRAINING -----------
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        seen = 0

        for xb, meta in tqdm(loaders["train"], desc=f"[Train] Epoch {epoch}/{epochs}"):
            x = xb.to(device, non_blocking=True)
            y = meta["class_id"].to(device, non_blocking=True)

            # build subject index tensor [B]
            subj_idx = torch.tensor(
                [subj2idx[s] for s in meta["subject"]],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, subj_idx)          # <-- multi-head forward
            loss = criterion(logits, y)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_acc += compute_accuracy(logits, y) * bs
            seen += bs

        train_loss /= seen
        train_acc /= seen

        # ----------- VALIDATION -----------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_seen = 0

        with torch.no_grad():
            for xb, meta in loaders["val"]:
                x = xb.to(device, non_blocking=True)
                y = meta["class_id"].to(device, non_blocking=True)
                subj_idx = torch.tensor(
                    [subj2idx[s] for s in meta["subject"]],
                    dtype=torch.long,
                    device=device,
                )

                logits = model(x, subj_idx)
                loss = criterion(logits, y)

                bs = x.size(0)
                val_loss += loss.item() * bs
                val_acc += compute_accuracy(logits, y) * bs
                val_seen += bs

        val_loss /= val_seen
        val_acc /= val_seen

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        # log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"Val loss={val_loss:.4f} acc={val_acc*100:.2f}%"
        )

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_dir / "best_multihead_model.pth")
            print(f"New best model at epoch {epoch} (val acc={val_acc*100:.2f}%)")

    # finish
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc
    history["best_state_dict"] = best_model_wts

    print(
        f"\nTraining complete. Best epoch={best_epoch} "
        f"val acc={best_val_acc*100:.2f}%"
    )

    return history