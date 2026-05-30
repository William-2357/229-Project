"""Standalone linear probing adapter for pretrained foundation EEG backbones.

Keeps the backbone permanently frozen; trains only a linear classification head.
For k=0: trains on source data features.
For k>0: trains on source features then refines on calibration features (backbone
still frozen throughout — this is pure linear probing, not fine-tuning).
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model
from models.foundations import FoundationBackbone, FoundationWithHead


class _HeadWrapper(nn.Module):
    """Thin nn.Module that applies a linear head to pre-extracted feature vectors.

    Allows train_epoch/evaluate_model to run on cached features without
    re-running the frozen backbone every epoch.
    """
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def train_linear_probe(
    model: FoundationWithHead,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    val_fraction: float,
) -> FoundationWithHead:
    """Train only the classification head with backbone frozen.

    Pre-extracts backbone features once so the epoch loop runs only the
    linear head — avoids redundant forward passes through the large frozen backbone.
    Mutates model.head in-place; returns model.
    """
    model.freeze_backbone()
    model.eval()

    feats = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[start: start + batch_size]).to(device)
            feats.append(model.backbone.get_features(xb).cpu().numpy())
    X_feat = np.concatenate(feats, axis=0)

    n_val = max(1, int(len(X_feat) * val_fraction))
    idx = np.random.permutation(len(X_feat))
    val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

    X_tr, y_tr = X_feat[train_idx], y[train_idx]
    X_val, y_val = X_feat[val_idx], y[val_idx]

    head = _HeadWrapper(model.head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc, best_state, patience_counter = -1.0, None, 0

    for _ in range(max_epochs):
        train_epoch(head, X_tr, y_tr, optimizer, device, batch_size)
        val_acc = evaluate_model(head, X_val, y_val, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    # head.head IS model.head (same object) — no copy needed
    return model


class LinearProbeAdapter(BaseAdapter):
    """Linear probing on a frozen pretrained foundation backbone.

    The backbone is never unfrozen. A linear classification head is trained
    on source features (k=0) and optionally refined on calibration features (k>0).
    Unlike FoundationFineTuneAdapter, this never touches backbone weights.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        lr_probe: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_probe: int = 100,
        patience_probe: int = 15,
        val_fraction_probe: float = 0.1,
        batch_size: int = 32,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"LinearProbeAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_probe = lr_probe
        self.weight_decay = weight_decay
        self.max_epochs_probe = max_epochs_probe
        self.patience_probe = patience_probe
        self.val_fraction_probe = val_fraction_probe
        self.batch_size = batch_size
        self._model: FoundationWithHead | None = None

    def _probe_kwargs(self) -> dict:
        return dict(
            device=self.device,
            batch_size=self.batch_size,
            lr=self.lr_probe,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs_probe,
            patience=self.patience_probe,
            val_fraction=self.val_fraction_probe,
        )

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
    ) -> "LinearProbeAdapter":
        if source_data is None:
            raise ValueError("LinearProbeAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)

        # Train head on source features; identical across K/repeat — cache it.
        cache_key = ("linear_probe", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
            model.freeze_backbone()
        else:
            model = train_linear_probe(model, X_src, y_src, **self._probe_kwargs())
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        # k>0: refine head on calibration features (backbone still frozen)
        if target_labeled is not None and len(target_labeled[0]) >= 2:
            X_cal, y_cal = target_labeled
            model = train_linear_probe(model, X_cal, y_cal, **self._probe_kwargs())

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
            for start in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[start: start + self.batch_size]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)
