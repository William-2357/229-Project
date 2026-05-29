"""Supervised full fine-tuning adapter.

Fine-tunes all model parameters on the target subject's K-minute
calibration set. AdamW with early stopping on 10% held-out calibration.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model


class FineTuneAdapter(BaseAdapter):
    """Full supervised fine-tune on target calibration data.

    Workflow:
        1. Pre-train on source subjects (LOSO)
        2. Fine-tune all parameters on target calibration set
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 # Source pre-training params
                 lr_src: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs_src: int = 200, batch_size: int = 64,
                 patience_src: int = 20, val_fraction_src: float = 0.1,
                 # Fine-tuning params
                 lr_ft: float = 1e-4, max_epochs_ft: int = 100,
                 patience_ft: int = 15, val_fraction_ft: float = 0.1):
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.batch_size = batch_size
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.lr_ft = lr_ft
        self.max_epochs_ft = max_epochs_ft
        self.patience_ft = patience_ft
        self.val_fraction_ft = val_fraction_ft
        self._model: nn.Module | None = None

    def _train_source(self, model: nn.Module, X_src: np.ndarray, y_src: np.ndarray) -> nn.Module:
        n_val = max(1, int(len(X_src) * self.val_fraction_src))
        idx = np.random.permutation(len(X_src))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X_src[train_idx], y_src[train_idx]
        X_val, y_val = X_src[val_idx], y_src[val_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_src, weight_decay=self.weight_decay)
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs_src):
            train_epoch(model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_src:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _finetune_target(self, model: nn.Module, X_cal: np.ndarray, y_cal: np.ndarray) -> nn.Module:
        if len(X_cal) < 2:
            return model

        n_val = max(1, int(len(X_cal) * self.val_fraction_ft))
        idx = np.random.permutation(len(X_cal))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        # Handle edge case: single-sample splits
        if len(train_idx) == 0:
            train_idx = val_idx

        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_ft, weight_decay=self.weight_decay)
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs_ft):
            train_epoch(model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_ft:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None) -> "FineTuneAdapter":
        if source_data is None:
            raise ValueError("FineTuneAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        model = self._clone_backbone().to(self.device)

        # Step 1: Source pre-training (cached by seed to avoid re-training across K values)
        # Original (no caching): model = self._train_source(model, X_src, y_src)
        cache_key = self.seed
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
        else:
            model = self._train_source(model, X_src, y_src)
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        # Step 2: Fine-tune on target calibration set (if provided)
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            model = self._finetune_target(model, X_cal, y_cal)

        self._model = model
        self._fit_time = time.time() - t0
        return self

    def _get_inference_model(self) -> nn.Module:
        return self._model if self._model is not None else self.backbone

    def predict(self, X: np.ndarray) -> np.ndarray:
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), 64):
                xb = torch.FloatTensor(X[start: start + 64]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)
