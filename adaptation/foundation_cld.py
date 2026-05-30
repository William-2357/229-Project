"""CLD and EA+CLD adapters for pretrained foundation EEG backbones.

Unlike CLDAdapter (which trains the backbone from scratch on source data),
these adapters treat the foundation backbone as a frozen feature extractor
and fit only the convex CLD head via ADMM.

For k=0:  CLD head is fit on source data features (zero-shot baseline).
For k>0:  CLD head is fit on labeled target calibration features only.
"""

import time
import numpy as np
import torch
import torch.nn as nn

import jax
import jax.numpy as jnp

from .base import BaseAdapter
from .cld import fit_cld_head
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from models.foundations import FoundationBackbone


def extract_foundation_features(
    backbone: FoundationBackbone,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run frozen foundation backbone on X and return (N, feature_dim) features."""
    backbone.eval()
    feats = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[start: start + batch_size]).to(device)
            feats.append(backbone.get_features(xb).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


class FoundationCLDAdapter(BaseAdapter):
    """CLD convex head on top of a frozen pretrained foundation backbone.

    The backbone is never fine-tuned — it is used purely as a feature
    extractor. Only the CLD head (fit via ADMM) is adapted per target subject.

    For k=0:  fit CLD on source features (zero-shot baseline).
    For k>0:  fit CLD on labeled target calibration features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        rank: int = 20,
        beta: float = 1e-3,
        rho: float = 0.01,
        gamma_ratio: float = 1.0,
        admm_iters: int = 50,
        pcg_iters: int = 10,
        n_neurons: int | None = None,
        batch_size: int = 32,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationCLDAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}. "
                f"Use CLDAdapter for specialist backbones."
            )
        super().__init__(backbone, device, seed)
        self.rank = rank
        self.beta = beta
        self.rho = rho
        self.gamma_ratio = gamma_ratio
        self.admm_iters = admm_iters
        self.pcg_iters = pcg_iters
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self._cld_model = None
        self._feat_mu: np.ndarray | None = None
        self._feat_sigma: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None) -> "FoundationCLDAdapter":
        if source_data is None:
            raise ValueError("FoundationCLDAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))
        n_neurons = self.n_neurons or (10 if n_classes == 2 else 32)

        backbone = self.backbone.to(self.device)

        if target_labeled is not None and len(target_labeled[0]) >= 2:
            X_fit, y_fit = target_labeled
        else:
            X_fit, y_fit = X_src, y_src

        X_feat = extract_foundation_features(backbone, X_fit, self.device, self.batch_size)

        # Use unlabeled target features for normalization statistics when available.
        # At small k the labeled set is tiny (e.g. 60 samples in 1024-d), making
        # per-feature mean/std noisy. Unlabeled target data is large regardless of k.
        if target_unlabeled is not None and len(target_unlabeled) >= 2:
            # Unlabeled target features are identical across all repeats (same X_unlabeled) — cache.
            tgt_cache_key = "foundation_cld_tgt_feats"
            if source_cache is not None and tgt_cache_key in source_cache:
                X_unlab = source_cache[tgt_cache_key]
            else:
                X_unlab = extract_foundation_features(
                    backbone, target_unlabeled, self.device, self.batch_size
                )
                if source_cache is not None:
                    source_cache[tgt_cache_key] = X_unlab
            norm_stats = (
                X_unlab.mean(axis=0, keepdims=True),
                X_unlab.std(axis=0, keepdims=True) + 1e-8,
            )
        else:
            norm_stats = None

        self._cld_model, self._feat_mu, self._feat_sigma = fit_cld_head(
            X_feat, y_fit, n_classes, n_neurons,
            self.rank, self.beta, self.rho, self.gamma_ratio,
            self.admm_iters, self.pcg_iters, self.seed,
            norm_stats=norm_stats,
        )

        self._fit_time = time.time() - t0
        return self

    def _predict_from_features(self, X_feat: np.ndarray) -> np.ndarray:
        X_norm = ((X_feat - self._feat_mu) / self._feat_sigma).astype(np.float32)
        return np.array(self._cld_model.stacked_predict(
            jnp.array(X_norm), self._cld_model.theta1, self._cld_model.theta2
        ))

    def _get_features(self, X: np.ndarray) -> np.ndarray:
        return extract_foundation_features(
            self.backbone.to(self.device), X, self.device, self.batch_size
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None:
            raise RuntimeError("FoundationCLDAdapter not fitted")
        return self._predict_from_features(self._get_features(X)).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None:
            raise RuntimeError("FoundationCLDAdapter not fitted")
        logits = self._predict_from_features(self._get_features(X))
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


class FoundationEACLDAdapter(FoundationCLDAdapter):
    """EA whitening + CLD convex head on a frozen foundation backbone.

    Applies Euclidean Alignment (He & Wu 2020) to raw EEG before passing
    through the foundation feature extractor.  EA reference is computed from
    unlabeled target trials; source data is aligned per-subject when
    source_per_subject is provided, otherwise globally.
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 epsilon: float = 1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None,
            source_per_subject: list | None = None) -> "FoundationEACLDAdapter":
        if source_data is None:
            raise ValueError("FoundationEACLDAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationEACLDAdapter requires target_unlabeled for EA")

        self._seed()

        X_src, y_src = source_data

        # EA whitening matrix from unlabeled target trials
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Align source — per-subject if available (correct per He & Wu 2020)
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))

        cal_aligned = None
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            cal_aligned = (euclidean_align(X_cal, self._target_R_inv_sqrt), y_cal)

        return super().fit(
            source_data=(X_src_aligned, y_src),
            target_unlabeled=euclidean_align(target_unlabeled, self._target_R_inv_sqrt),
            target_labeled=cal_aligned,
            source_cache=source_cache,
        )

    def _align_target(self, X: np.ndarray) -> np.ndarray:
        return euclidean_align(X, self._target_R_inv_sqrt)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEACLDAdapter not fitted")
        return super().predict(self._align_target(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEACLDAdapter not fitted")
        return super().predict_proba(self._align_target(X))
