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
import numpy as np
import torch

from .base import train_epoch, evaluate_model
from models.foundations import FoundationBackbone, FoundationWithHead


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
