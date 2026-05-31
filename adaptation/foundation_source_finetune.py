"""Source fine-tuning for foundation EEG backbones (reconstructed shared infra).

NOTE: the original `foundation_source_finetune.py` was gitignored by a teammate and
never committed (its imports survive in foundation_sft_finetune.py /
foundation_source_cld_old.py / run_experiment.py). This reconstruction restores the
shared `build_source_finetuned_foundation_model` so every `foundation_sft_*` method —
and the convex research method — starts from the *same* source-task-shaped backbone.

What it does:
  1. wrap a pretrained foundation encoder in a linear head (FoundationWithHead)
  2. unfreeze and fine-tune backbone + head on pooled source subjects (early stopping)
  3. return the trained model (caller freezes it and adapts a head per target)

Disk cache: the source fine-tune is the expensive step (full backbone FT) and is
identical for a given (backbone, source-split, seed, FT-hyperparams). We cache it under
data/sft_checkpoints/ so it runs once per LOSO fold and is reused across the whole
autoresearch loop and across baselines — making the FM comparison both fast and fair.
"""

from __future__ import annotations

import os
import copy
import hashlib
import numpy as np
import torch

from .base import train_epoch, evaluate_model
from models.foundations import FoundationBackbone, FoundationWithHead

# Local cache root (overridable via env). One file per (backbone, src-split, seed, cfg).
CACHE_ROOT = os.environ.get("SFT_CACHE_DIR", os.path.join(os.getcwd(), "data", "sft_checkpoints"))


def _source_split_hash(X_src: np.ndarray, y_src: np.ndarray) -> str:
    """Stable id for a LOSO source split (which subjects are pooled in)."""
    h = hashlib.md5()
    h.update(X_src.shape.__repr__().encode())
    h.update(X_src.tobytes()[:100000])      # prefix is enough to disambiguate folds
    h.update(y_src.tobytes())
    return h.hexdigest()[:10]


def _cache_path(backbone, seed, X_src, y_src, cfg: dict) -> str:
    name = backbone.__class__.__name__.lower()
    cfg_sig = hashlib.md5(repr(sorted(cfg.items())).encode()).hexdigest()[:6]
    split = _source_split_hash(X_src, y_src)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    return os.path.join(CACHE_ROOT, f"{name}_seed{seed}_src{split}_cfg{cfg_sig}_sft.pt")


def build_source_finetuned_foundation_model(
    backbone: FoundationBackbone,
    n_classes: int,
    X_src: np.ndarray,
    y_src: np.ndarray,
    *,
    device,
    lr_src: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs_src: int = 200,
    patience_src: int = 25,
    val_fraction_src: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
    use_disk_cache: bool = True,
) -> FoundationWithHead:
    """Full source fine-tune of a foundation backbone + linear head.

    Returns a FoundationWithHead trained on (X_src, y_src). Disk-cached per
    (backbone, source-split, seed, FT-config).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    cfg = dict(lr_src=lr_src, weight_decay=weight_decay, max_epochs_src=max_epochs_src,
               patience_src=patience_src, val_fraction_src=val_fraction_src,
               batch_size=batch_size, n_classes=n_classes)

    model = FoundationWithHead(copy.deepcopy(backbone), n_classes).to(device)

    path = _cache_path(backbone, seed, X_src, y_src, cfg) if use_disk_cache else None
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return model

    # Full fine-tune: backbone + head on source.
    model.unfreeze_backbone()
    n_val = max(1, int(len(X_src) * val_fraction_src))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_src))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X_src[train_idx], y_src[train_idx]
    X_val, y_val = X_src[val_idx], y_src[val_idx]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_src, weight_decay=weight_decay)
    best_val_acc, best_state, patience_counter = -1.0, None, 0
    for _ in range(max_epochs_src):
        train_epoch(model, X_tr, y_tr, optimizer, device, batch_size)
        val_acc = evaluate_model(model, X_val, y_val, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_src:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    if path:
        torch.save(model.state_dict(), path)
    return model
