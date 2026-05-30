"""Euclidean Alignment adapter for pretrained foundation EEG backbones.

Mirrors EAAdapter but replaces backbone source-training with a linear probe
on the frozen pretrained foundation backbone.

K=0 unsupervised method: requires only unlabeled target trials for EA alignment.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .linear_probe import train_linear_probe
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationEAAdapter(BaseAdapter):
    """EA alignment + linear probe on a frozen pretrained foundation backbone.

    Workflow:
        1. Align source subjects (per-subject if source_per_subject provided)
        2. Align target using unlabeled target trials
        3. Train linear head on aligned source features (backbone frozen)
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
        epsilon: float = 1e-6,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationEAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_probe = lr_probe
        self.weight_decay = weight_decay
        self.max_epochs_probe = max_epochs_probe
        self.patience_probe = patience_probe
        self.val_fraction_probe = val_fraction_probe
        self.batch_size = batch_size
        self.epsilon = epsilon
        self._model: FoundationWithHead | None = None
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
        source_per_subject: list | None = None,
    ) -> "FoundationEAAdapter":
        if source_data is None:
            raise ValueError("FoundationEAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationEAAdapter requires target_unlabeled for EA alignment")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Step 1: Align source — per-subject if available (correct per He & Wu 2020)
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))

        # Step 2: Compute target alignment transform from unlabeled target trials
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Step 3: Linear probe on aligned source (backbone frozen)
        # EA alignment changes input distribution, so the probe must be re-trained
        # per subject (cannot reuse the unaligned source_cache from other adapters).
        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)

        cache_key = ("foundation_ea_probe", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
            model.freeze_backbone()
        else:
            model = train_linear_probe(
                model, X_src_aligned, y_src,
                device=self.device,
                batch_size=self.batch_size,
                lr=self.lr_probe,
                weight_decay=self.weight_decay,
                max_epochs=self.max_epochs_probe,
                patience=self.patience_probe,
                val_fraction=self.val_fraction_probe,
            )
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        self._model = model
        self._fit_time = time.time() - t0
        return self

    def _align_target(self, X: np.ndarray) -> np.ndarray:
        return euclidean_align(X, self._target_R_inv_sqrt)

    def _get_inference_model(self) -> nn.Module:
        return self._model if self._model is not None else self.backbone

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEAAdapter not fitted")
        X_aligned = self._align_target(X)
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_aligned), self.batch_size):
                xb = torch.FloatTensor(X_aligned[start: start + self.batch_size]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEAAdapter not fitted")
        return super().predict_proba(self._align_target(X))
