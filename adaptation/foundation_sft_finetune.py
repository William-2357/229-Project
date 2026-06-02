"""Source-finetuned foundation backbone + target fine-tuning.

Closes the architecture gap between foundation and specialist models:
  1. Start from a pretrained foundation backbone
  2. Fine-tune the full backbone + head on pooled source subjects
  3. Freeze the source-task-shaped backbone
  4. K>0: unfreeze and fine-tune on target calibration data

This mirrors the specialist FineTuneAdapter but starts from pretrained
foundation weights rather than random initialization.
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model
from .foundation_source_finetune import load_or_build_sft_model
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationSFTFineTuneAdapter(BaseAdapter):
    """Source-finetuned foundation backbone + target fine-tuning.

    K=0: source fine-tune → freeze → evaluate (zero-shot, identical to LOSO)
    K>0: source fine-tune → freeze → unfreeze → fine-tune on target calibration
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        # Source fine-tuning params
        lr_src: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_src: int = 200,
        patience_src: int = 25,
        val_fraction_src: float = 0.1,
        batch_size: int = 32,
        # Target fine-tuning params — low LR to avoid overwriting source representations
        lr_tgt: float = 1e-5,
        max_epochs_tgt: int = 100,
        patience_tgt: int = 15,
        val_fraction_tgt: float = 0.1,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationSFTFineTuneAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.batch_size = batch_size
        self.lr_tgt = lr_tgt
        self.max_epochs_tgt = max_epochs_tgt
        self.patience_tgt = patience_tgt
        self.val_fraction_tgt = val_fraction_tgt
        self._model: FoundationWithHead | None = None

    def _source_ft_kwargs(self) -> dict:
        return dict(
            device=self.device,
            lr_src=self.lr_src,
            weight_decay=self.weight_decay,
            max_epochs_src=self.max_epochs_src,
            patience_src=self.patience_src,
            val_fraction_src=self.val_fraction_src,
            batch_size=self.batch_size,
        )

    def _finetune_target(
        self, model: FoundationWithHead, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> FoundationWithHead:
        if len(X_cal) < 2:
            return model

        model.unfreeze_backbone()

        n_val = max(1, int(len(X_cal) * self.val_fraction_tgt))
        idx = np.random.permutation(len(X_cal))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr_tgt, weight_decay=self.weight_decay
        )
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs_tgt):
            train_epoch(model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_tgt:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
    ) -> "FoundationSFTFineTuneAdapter":
        if source_data is None:
            raise ValueError("FoundationSFTFineTuneAdapter requires source_data")

        self._seed()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Source fine-tune — one-time, shared source-side cost; excluded from
        # fit_time so the timer measures only per-K target adaptation. Reuses a
        # disk checkpoint shared across ALL SFT methods/jobs/runs (see
        # load_or_build_sft_model), so the ~3-4 min fine-tune isn't recomputed.
        model = load_or_build_sft_model(
            self.backbone, n_classes, X_src, y_src,
            seed=self.seed, source_cache=source_cache, **self._source_ft_kwargs(),
        )
        model.freeze_backbone()

        # ---- fit_time covers target adaptation only (frozen backbone is ready) ----
        t0 = time.time()
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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        probs = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[start: start + self.batch_size]).to(self.device)
                probs.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
        return np.concatenate(probs)
