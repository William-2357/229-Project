"""Zero-shot LOSO baseline for pretrained foundation EEG backbones.

Freezes the backbone and trains a linear head on source data only.
No target data is used — this is the zero-shot transfer floor for foundation models.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter
from .linear_probe import train_linear_probe
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationLOSOAdapter(BaseAdapter):
    """Frozen backbone + linear head trained on source data only.

    Analogous to LOSOAdapter for specialist models, but skips source training
    from scratch — the pretrained backbone already provides transferable features.
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
                f"FoundationLOSOAdapter requires a FoundationBackbone, "
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
    ) -> "FoundationLOSOAdapter":
        if source_data is None:
            raise ValueError("FoundationLOSOAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)

        cache_key = ("foundation_loso_probe", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
            model.freeze_backbone()
        else:
            model = train_linear_probe(model, X_src, y_src, **self._probe_kwargs())
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        # target data intentionally ignored — this is the zero-shot floor
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
