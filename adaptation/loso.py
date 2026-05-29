"""Zero-shot LOSO (Leave-One-Subject-Out) baseline adapter."""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model


class LOSOAdapter(BaseAdapter):
    """Train on pooled source subjects, evaluate directly on target (no adaptation).

    This is the zero-shot floor: K=0, no target data used.
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 lr: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 200,
                 batch_size: int = 64, patience: int = 20, val_fraction: float = 0.1):
        super().__init__(backbone, device, seed)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self._model: nn.Module | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None) -> "LOSOAdapter":
        if source_data is None:
            raise ValueError("LOSOAdapter requires source_data (X, y)")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        model = self._clone_backbone().to(self.device)

        # Hold out val_fraction of source for early stopping
        n_val = max(1, int(len(X_src) * self.val_fraction))
        idx = np.random.permutation(len(X_src))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr, y_tr = X_src[train_idx], y_src[train_idx]
        X_val, y_val = X_src[val_idx], y_src[val_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_acc = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            train_epoch(model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
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
