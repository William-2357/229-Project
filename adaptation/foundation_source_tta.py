"""T3A test-time adaptation after source fine-tuning a foundation backbone.

Unlike FoundationTTAAdapter (frozen pretrained backbone + linear probe on source),
this adapter first fine-tunes the full backbone on source data, giving the model
a task-shaped representation before T3A prototype adjustment.

Workflow:
    1. Fine-tune full backbone + head on pooled source data
    2. Freeze the source-task-shaped backbone
    3. T3A: adjust head prototypes using high-confidence unlabeled target predictions
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter
from .foundation_tta import _run_foundation_t3a
from .foundation_source_finetune import load_or_build_sft_model
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationSFTTTAAdapter(BaseAdapter):
    """T3A test-time adaptation after source fine-tuning of a foundation backbone."""

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
        t3a_confidence: float = 0.9,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationSFTTTAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.batch_size = batch_size
        self.t3a_confidence = t3a_confidence
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
    ) -> "FoundationSFTTTAAdapter":
        if source_data is None:
            raise ValueError("FoundationSFTTTAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationSFTTTAAdapter requires target_unlabeled for T3A")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Step 1: Source fine-tune — shared disk checkpoint across ALL foundation
        # SFT methods (see load_or_build_sft_model), reused across jobs/runs.
        model = load_or_build_sft_model(
            self.backbone, n_classes, X_src, y_src,
            seed=self.seed, source_cache=source_cache, **self._source_ft_kwargs(),
        )

        # Step 2: Freeze backbone — T3A only adjusts head prototypes
        model.freeze_backbone()

        # Step 3: T3A prototype adjustment from high-confidence unlabeled target predictions
        model = _run_foundation_t3a(
            model, target_unlabeled, self.device,
            n_classes=n_classes,
            confidence_threshold=self.t3a_confidence,
            batch_size=self.batch_size,
        )

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
