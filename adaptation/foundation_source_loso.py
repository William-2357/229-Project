"""Zero-shot LOSO and EA baselines for source-finetuned foundation backbones.

These adapters fill the gap between:
  - foundation_loso/ea: frozen pretrained backbone + linear probe on source
  - specialist loso/ea: backbone trained end-to-end on source data

Workflow for both:
  1. Fine-tune full foundation backbone + head on pooled source data
  2. Freeze the resulting source-task-shaped backbone
  3. Evaluate on target without any target-side adaptation (LOSO)
     or after EA alignment of source and target (EA variant)
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_source_finetune import build_source_finetuned_foundation_model
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationSFTLOSOAdapter(BaseAdapter):
    """Source-finetuned foundation backbone as K=0 zero-shot baseline.

    Unlike FoundationLOSOAdapter (frozen backbone + linear probe on source),
    this unfreezes the full backbone and fine-tunes it on source data — matching
    the specialist LOSO baseline. No target-side adaptation is applied.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        lr_src: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_src: int = 200,
        patience_src: int = 25,
        val_fraction_src: float = 0.1,
        batch_size: int = 32,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationSFTLOSOAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.batch_size = batch_size
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

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
    ) -> "FoundationSFTLOSOAdapter":
        if source_data is None:
            raise ValueError("FoundationSFTLOSOAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        cache_key = ("foundation_sft_loso_source_ft", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
        else:
            model = build_source_finetuned_foundation_model(
                self.backbone, n_classes, X_src, y_src, **self._source_ft_kwargs()
            )
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        model.freeze_backbone()
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


class FoundationSFTEAAdapter(FoundationSFTLOSOAdapter):
    """EA whitening + source-finetuned foundation backbone, K=0 zero-shot baseline.

    Workflow:
        1. EA-align source subjects (per-subject if source_per_subject provided)
        2. Fine-tune full backbone + head on aligned source data
        3. EA-align target using unlabeled target trials
        4. Evaluate on aligned target (no further target-side adaptation)
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 epsilon: float = 1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
        source_per_subject: list | None = None,
    ) -> "FoundationSFTEAAdapter":
        if source_data is None:
            raise ValueError("FoundationSFTEAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationSFTEAAdapter requires target_unlabeled for EA alignment")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # EA-align source — per-subject is the correct approach (He & Wu 2020)
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))

        # EA-align target from unlabeled trials
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Source fine-tune on aligned source (cached — alignment is fixed per subject)
        cache_key = ("foundation_sft_ea_source_ft", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
        else:
            model = build_source_finetuned_foundation_model(
                self.backbone, n_classes, X_src_aligned, y_src, **self._source_ft_kwargs()
            )
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        model.freeze_backbone()
        self._model = model
        self._fit_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSFTEAAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSFTEAAdapter not fitted")
        return super().predict_proba(euclidean_align(X, self._target_R_inv_sqrt))
