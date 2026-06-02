"""Shared helpers for source-side fine-tuning of foundation EEG backbones.

These utilities implement the missing middle ground between:
  - frozen pretrained foundation features
  - training a specialist model from scratch

Workflow:
  1. Start from a pretrained FoundationBackbone
  2. Add a linear task head
  3. Fine-tune the full model on pooled source-subject data
  4. Reuse the resulting source-task-shaped backbone for lightweight target adaptation
"""

from __future__ import annotations

import copy
import hashlib
import os
import numpy as np
import torch

from .base import train_epoch, evaluate_model
from models.foundations import FoundationBackbone, FoundationWithHead


# Canonical on-disk location for source-fine-tuned backbones (Modal volume).
SFT_CHECKPOINT_DIR = "/data/sft_checkpoints"


def build_source_finetuned_foundation_model(
    backbone: FoundationBackbone,
    n_classes: int,
    X_src: np.ndarray,
    y_src: np.ndarray,
    device: torch.device,
    lr_src: float,
    weight_decay: float,
    max_epochs_src: int,
    patience_src: int,
    val_fraction_src: float,
    batch_size: int,
) -> FoundationWithHead:
    """Fine-tune a pretrained foundation backbone + head on source data.

    Returns the best validation checkpoint. Unlike the linear-probe helper, this
    unfreezes the full backbone so the pretrained encoder becomes source-task-shaped.
    """
    model = FoundationWithHead(copy.deepcopy(backbone), n_classes).to(device)
    model.unfreeze_backbone()

    n_val = max(1, int(len(X_src) * val_fraction_src))
    idx = np.random.permutation(len(X_src))
    val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

    X_tr, y_tr = X_src[train_idx], y_src[train_idx]
    X_val, y_val = X_src[val_idx], y_src[val_idx]

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_src, weight_decay=weight_decay
    )
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
    return model


def _commit_data_volume() -> None:
    """Persist freshly written checkpoints to the Modal volume (no-op off-Modal)."""
    try:
        import __main__
        if hasattr(__main__, "data_volume"):
            __main__.data_volume.commit()
    except Exception:
        pass


def sft_checkpoint_path(
    backbone: FoundationBackbone,
    seed: int,
    X_src: np.ndarray,
    y_src: np.ndarray,
    lr_src: float,
    max_epochs_src: int,
    checkpoint_dir: str = SFT_CHECKPOINT_DIR,
) -> str:
    """Canonical disk path for a source-fine-tuned backbone.

    Shared by ALL SFT adapters (finetune / lora / cld / anchored) so the ~3-4 min
    source fine-tune is computed once per (backbone, LOSO fold, lr, epochs) and
    reused across methods, jobs and runs. The fold is identified by a hash of the
    pooled source data; lr/epochs are the only SFT hyperparameters that vary in
    practice (weight_decay/patience/val_fraction/batch_size are fixed defaults that
    are identical across all adapters).
    """
    backbone_name = backbone.__class__.__name__.lower()
    src_hash = hashlib.md5(X_src.tobytes()[:50000] + y_src.tobytes()).hexdigest()[:8]
    lr_tag = f"lr{lr_src:.0e}".replace("-", "n")
    return os.path.join(
        checkpoint_dir,
        f"{backbone_name}_seed{seed}_src_{src_hash}_{lr_tag}_ep{max_epochs_src}_sft.pt",
    )


def load_or_build_sft_model(
    backbone: FoundationBackbone,
    n_classes: int,
    X_src: np.ndarray,
    y_src: np.ndarray,
    *,
    seed: int,
    source_cache: dict | None = None,
    in_memory_key: str = "sft_model_state",
    checkpoint_dir: str = SFT_CHECKPOINT_DIR,
    **sft_kwargs,
) -> FoundationWithHead:
    """Return a source-fine-tuned FoundationWithHead, reusing a cached result.

    Lookup order:
      1. in-memory ``source_cache`` (same job, across K/repeats)
      2. shared disk checkpoint (across jobs and runs) — see sft_checkpoint_path
      3. otherwise fine-tune from scratch and persist for everyone else.

    Because the disk checkpoint is shared across all SFT methods (which use
    identical SFT hyperparameters and a seeded, deterministic split), the
    expensive source fine-tune happens at most once per fold instead of once per
    (method, fold). ``sft_kwargs`` must contain the build_source_finetuned_*
    arguments (device, lr_src, weight_decay, max_epochs_src, patience_src,
    val_fraction_src, batch_size).
    """
    device = sft_kwargs["device"]

    # 1) in-memory (same job)
    if source_cache is not None and in_memory_key in source_cache:
        model = FoundationWithHead(copy.deepcopy(backbone), n_classes).to(device)
        model.load_state_dict(copy.deepcopy(source_cache[in_memory_key]))
        return model

    # 2) shared disk checkpoint (across jobs and runs)
    path = None
    if os.path.isdir("/data"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = sft_checkpoint_path(
            backbone, seed, X_src, y_src,
            sft_kwargs["lr_src"], sft_kwargs["max_epochs_src"], checkpoint_dir,
        )

    if path is not None and os.path.exists(path):
        model = FoundationWithHead(copy.deepcopy(backbone), n_classes).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        # 3) fine-tune, then persist so other methods/jobs/runs reuse it
        model = build_source_finetuned_foundation_model(
            backbone, n_classes, X_src, y_src, **sft_kwargs
        )
        if path is not None:
            torch.save(model.state_dict(), path)
            _commit_data_volume()

    if source_cache is not None:
        source_cache[in_memory_key] = copy.deepcopy(model.state_dict())
    return model
