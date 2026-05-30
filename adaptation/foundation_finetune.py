"""Supervised full fine-tuning adapter for pretrained foundation EEG backbones.

Unlike FineTuneAdapter (which trains from scratch on source data), this adapter
treats the foundation backbone as a starting point and skips source pre-training.

For k=0:  trains only the linear classification head on source features
          (linear probe — fast, avoids overwriting pretrained representations).
For k>0:  unfreezes the full backbone and fine-tunes everything on the
          calibration set with a low learning rate to prevent catastrophic
          forgetting.
"""

import time
import copy
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


class FoundationFineTuneAdapter(BaseAdapter):
    """Full fine-tuning of a pretrained foundation backbone + linear head.

    Workflow:
        k=0: freeze backbone, train linear head on source features (linear probe)
        k>0: unfreeze backbone, fine-tune all parameters on calibration set
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        # Linear probe (k=0) params
        lr_probe: float = 1e-3,
        max_epochs_probe: int = 100,
        patience_probe: int = 15,
        val_fraction_probe: float = 0.1,
        # Full fine-tune (k>0) params — low LR to avoid catastrophic forgetting
        lr_ft: float = 1e-5,
        weight_decay: float = 1e-4,
        max_epochs_ft: int = 100,
        patience_ft: int = 15,
        val_fraction_ft: float = 0.1,
        batch_size: int = 32,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationFineTuneAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_probe = lr_probe
        self.max_epochs_probe = max_epochs_probe
        self.patience_probe = patience_probe
        self.val_fraction_probe = val_fraction_probe
        self.lr_ft = lr_ft
        self.weight_decay = weight_decay
        self.max_epochs_ft = max_epochs_ft
        self.patience_ft = patience_ft
        self.val_fraction_ft = val_fraction_ft
        self.batch_size = batch_size
        self._model: FoundationWithHead | None = None

    def _train_linear_probe(
        self, model: FoundationWithHead, X: np.ndarray, y: np.ndarray
    ) -> FoundationWithHead:
        """Train only the classification head with backbone frozen.

        Pre-extracts backbone features once so the epoch loop runs only the
        linear head — avoids 100x redundant forward passes through the large
        frozen backbone.
        """
        model.freeze_backbone()

        # Extract features once (backbone frozen, so features are constant)
        model.eval()
        feats = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[start: start + self.batch_size]).to(self.device)
                feats.append(model.backbone.get_features(xb).cpu().numpy())
        X_feat = np.concatenate(feats, axis=0)

        n_val = max(1, int(len(X_feat) * self.val_fraction_probe))
        idx = np.random.permutation(len(X_feat))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_feat[train_idx], y[train_idx]
        X_val, y_val = X_feat[val_idx], y[val_idx]

        # Train only the head on cached features
        head = _HeadWrapper(model.head)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=self.lr_probe, weight_decay=self.weight_decay
        )
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs_probe):
            train_epoch(head, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(head, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(head.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_probe:
                break

        if best_state is not None:
            head.load_state_dict(best_state)
        # head.head IS model.head (same object) — no copy needed
        return model

    def _finetune_target(
        self, model: FoundationWithHead, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> FoundationWithHead:
        """Unfreeze backbone and fine-tune all parameters on calibration data."""
        if len(X_cal) < 2:
            return model

        model.unfreeze_backbone()

        n_val = max(1, int(len(X_cal) * self.val_fraction_ft))
        idx = np.random.permutation(len(X_cal))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr_ft, weight_decay=self.weight_decay
        )
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

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
    ) -> "FoundationFineTuneAdapter":
        if source_data is None:
            raise ValueError("FoundationFineTuneAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Build backbone + head; backbone starts frozen (from build_foundation_model)
        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)

        # k=0: linear probe on source features
        # k>0: linear probe first (warm up head), then full fine-tune on calibration
        # Probe result is identical across all K/repeat calls (same X_src + seed) — cache it.
        cache_key = ("foundation_finetune_probe", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
        else:
            model = self._train_linear_probe(model, X_src, y_src)
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        if target_labeled is not None and len(target_labeled[0]) >= 2:
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
            for start in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[start: start + self.batch_size]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)
