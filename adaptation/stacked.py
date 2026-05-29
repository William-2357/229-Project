"""Stacked adaptation: EA preprocessing + CLD convex head.

EACLDAdapter applies Euclidean Alignment whitening (unsupervised,
from unlabeled target trials) before fitting the CLD head. This
combines the covariance-based domain alignment of EA with the
sample-efficient convex classifier of CLD.
"""

import time
import copy
import numpy as np
import torch.nn as nn

from .base import BaseAdapter
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .cld import CLDAdapter


class EACLDAdapter(BaseAdapter):
    """EA whitening + CLD convex head (stacked).

    Workflow:
        1. Compute EA transform from unlabeled target trials
        2. Align source data per-subject (He & Wu 2020) and target data
        3. Run CLDAdapter on the aligned features
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 epsilon: float = 1e-6,
                 lr_src: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs_src: int = 200, batch_size: int = 64,
                 patience_src: int = 20, val_fraction_src: float = 0.1,
                 rank: int = 20, beta: float = 1e-3, rho: float = 0.01,
                 gamma_ratio: float = 1.0, admm_iters: int = 50,
                 pcg_iters: int = 10, n_neurons: int | None = None):
        super().__init__(backbone, device, seed)
        self.epsilon = epsilon
        self._cld_kwargs = dict(
            lr_src=lr_src, weight_decay=weight_decay,
            max_epochs_src=max_epochs_src, batch_size=batch_size,
            patience_src=patience_src, val_fraction_src=val_fraction_src,
            rank=rank, beta=beta, rho=rho, gamma_ratio=gamma_ratio,
            admm_iters=admm_iters, pcg_iters=pcg_iters, n_neurons=n_neurons,
        )
        self._cld_adapter: CLDAdapter | None = None
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None,
            source_per_subject: list | None = None) -> "EACLDAdapter":
        if source_data is None:
            raise ValueError("EACLDAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("EACLDAdapter requires target_unlabeled for EA")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data

        # Step 1: EA whitening matrix from unlabeled target
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Step 2: Align source per-subject (correct per He & Wu 2020)
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            X_src_aligned = euclidean_align(
                X_src, matrix_sqrt_inv(compute_mean_covariance(X_src, self.epsilon))
            )

        X_unlabeled_aligned = euclidean_align(target_unlabeled, self._target_R_inv_sqrt)

        cal_aligned = None
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            cal_aligned = (euclidean_align(X_cal, self._target_R_inv_sqrt), y_cal)

        # Step 3: CLDAdapter on aligned data
        self._cld_adapter = CLDAdapter(
            backbone=copy.deepcopy(self.backbone),
            device=str(self.device),
            seed=self.seed,
            **self._cld_kwargs,
        )
        self._cld_adapter.fit(
            source_data=(X_src_aligned, y_src),
            target_unlabeled=X_unlabeled_aligned,
            target_labeled=cal_aligned,
            source_cache=source_cache,
        )

        self._fit_time = time.time() - t0
        return self

    def _align_target(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            return X
        return euclidean_align(X, self._target_R_inv_sqrt)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._cld_adapter.predict(self._align_target(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._cld_adapter.predict_proba(self._align_target(X))
