"""Source-finetuned LoRA adapters for foundation EEG backbones (reconstructed).

NOTE: original `foundation_source_lora.py` was gitignored by a teammate and never
committed (imported by run_experiment.py). Reconstructed so the `foundation_sft_lora` /
`foundation_sft_ea_lora` LoRA baselines run locally on the SAME source-fine-tuned
backbone as the convex method — the fair bar the convex head must beat.

Flow (the foundation analogue of specialist LoRA):
  1. source-fine-tune backbone + head on pooled source (shared infra, disk-cached)
  2. freeze backbone
  3. K=0: return the source-FT model (LOSO / zero-shot)
  4. K>0: attach LoRA to the source-shaped backbone, fine-tune LoRA params + head
          on the K-minute calibration set
"""

from __future__ import annotations

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from .base import BaseAdapter, train_epoch, evaluate_model
from .foundation_lora import _get_lora_target_modules
from .foundation_source_finetune import build_source_finetuned_foundation_model
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationSourceFineTuneLoRAAdapter(BaseAdapter):
    """Source-finetuned foundation backbone + LoRA calibration."""

    def __init__(
        self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
        # source fine-tuning (shared with all sft_* methods)
        lr_src: float = 1e-3, weight_decay: float = 1e-4, max_epochs_src: int = 200,
        patience_src: int = 25, val_fraction_src: float = 0.1, batch_size: int = 32,
        # LoRA calibration
        rank: int = 8, lr_lora: float = 1e-3, max_epochs_lora: int = 100,
        patience_lora: int = 15, val_fraction_ft: float = 0.1,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError("FoundationSourceFineTuneLoRAAdapter requires a FoundationBackbone")
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.batch_size = batch_size
        self.rank = rank
        self.lr_lora = lr_lora
        self.max_epochs_lora = max_epochs_lora
        self.patience_lora = patience_lora
        self.val_fraction_ft = val_fraction_ft
        self._model: nn.Module | None = None

    def _source_ft_kwargs(self) -> dict:
        return dict(device=self.device, lr_src=self.lr_src, weight_decay=self.weight_decay,
                    max_epochs_src=self.max_epochs_src, patience_src=self.patience_src,
                    val_fraction_src=self.val_fraction_src, batch_size=self.batch_size,
                    seed=self.seed)

    def _source_ft(self, X_src, y_src, n_classes, source_cache):
        cache_key = ("sft_lora_source_ft", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
            return model
        model = build_source_finetuned_foundation_model(
            self.backbone, n_classes, X_src, y_src, **self._source_ft_kwargs())
        if source_cache is not None:
            source_cache[cache_key] = copy.deepcopy(model.state_dict())
        return model

    def _finetune_lora(self, lora_model, X_cal, y_cal):
        if len(X_cal) < 2:
            return lora_model
        n_val = max(1, int(len(X_cal) * self.val_fraction_ft))
        idx = np.random.permutation(len(X_cal))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx
        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]
        trainable = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr_lora, weight_decay=self.weight_decay)
        best_val_acc, best_state, patience = -1.0, None, 0
        for _ in range(self.max_epochs_lora):
            train_epoch(lora_model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(lora_model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc, best_state, patience = val_acc, copy.deepcopy(lora_model.state_dict()), 0
            else:
                patience += 1
                if patience >= self.patience_lora:
                    break
        if best_state is not None:
            lora_model.load_state_dict(best_state)
        return lora_model

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        if source_data is None:
            raise ValueError("FoundationSourceFineTuneLoRAAdapter requires source_data")
        self._seed()
        t0 = time.time()
        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        model = self._source_ft(X_src, y_src, n_classes, source_cache)
        model.freeze_backbone()

        if target_labeled is None or len(target_labeled[0]) < 2:
            self._model = model                      # K=0: LOSO
            self._fit_time = time.time() - t0
            return self

        target_modules = _get_lora_target_modules(model, rank=self.rank)
        if not target_modules:
            self._model = model
            self._fit_time = time.time() - t0
            return self
        lora_config = LoraConfig(r=self.rank, lora_alpha=self.rank * 2,
                                 target_modules=target_modules, lora_dropout=0.1, bias="none")
        lora_model = get_peft_model(model, lora_config)
        self._model = self._finetune_lora(lora_model, *target_labeled)
        self._fit_time = time.time() - t0
        return self

    def _infer(self, X):
        model = self._model if self._model is not None else self.backbone
        model.eval(); model.to(self.device)
        preds = []
        with torch.no_grad():
            for s in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[s:s + self.batch_size]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._infer(X)


class FoundationSourceFineTuneEALoRAAdapter(FoundationSourceFineTuneLoRAAdapter):
    """EA whitening + source-finetuned backbone + LoRA calibration."""

    def __init__(self, backbone, device="cpu", seed=42, epsilon=1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._R: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache=None, source_per_subject=None):
        if target_unlabeled is None:
            raise ValueError("EA-LoRA requires target_unlabeled")
        X_src, y_src = source_data
        self._R = matrix_sqrt_inv(compute_mean_covariance(target_unlabeled, self.epsilon))
        X_src_a = euclidean_align(X_src, matrix_sqrt_inv(compute_mean_covariance(X_src, self.epsilon)))
        cal = None
        if target_labeled is not None:
            cal = (euclidean_align(target_labeled[0], self._R), target_labeled[1])
        return super().fit((X_src_a, y_src), euclidean_align(target_unlabeled, self._R),
                           cal, source_cache=source_cache)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(euclidean_align(X, self._R) if self._R is not None else X)
