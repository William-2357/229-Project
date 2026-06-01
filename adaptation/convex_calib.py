"""Convex calibration adapter — the ONE file the autoresearch loop edits.

Research surface for "convex NN for low-resource EEG calibration" (research/program.md).
Backbone: a source-fine-tuned MIRepNet foundation encoder (frozen after source-FT),
exactly like the `foundation_sft_*` baselines, so the comparison vs sft_lora /
sft_finetune is apples-to-apples — the only moving part is the adaptation head.

The convex head (jaxcld CVX_ReLU_MLP + ADMM) has provable global optimality / margin
stability; the goal is to beat LoRA in the low-K calibration regime.

Iteration 1 — kill the low-K dip
--------------------------------
The stock `foundation_sft_cld` refits the convex head FROM SCRATCH on only the ~12
calibration trials at K>0, throwing away everything the source-shaped backbone+head
knew → BCA collapses below LOSO at K=0.5 (the dip in mirepnet_performance.png). Fix:
fit the convex head on the UNION of (subsampled) source features + UPWEIGHTED
calibration features. One global convex solve, well-posed at every K. K=0 → source-only
(LOSO head, no dip); K>0 → boundary nudged toward target. `cal_balance` sets how much
total mass the calibration trials get relative to the source pool.
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

import jax.numpy as jnp

from .base import BaseAdapter
from .cld import fit_cld_head, maybe_reduce_features
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_cld import extract_foundation_features
from .foundation_source_finetune import build_source_finetuned_foundation_model
from models.foundations import FoundationBackbone

# ---------------------------------------------------------------------------
# HPARAMS — the loop's primary tuning surface. Keep flat and documented.
# ---------------------------------------------------------------------------
HPARAMS = dict(
    # --- source fine-tuning of the FM backbone (shared with baselines; disk-cached) ---
    lr_src=1e-3, weight_decay=1e-4, max_epochs_src=200, patience_src=25,
    val_fraction_src=0.1, ft_batch_size=32,

    # --- front-end ---
    use_ea=False,           # EA whitening of raw EEG (MIRepNet already EA-normalizes internally)
    ea_epsilon=1e-6,

    # --- convex ReLU head (jaxcld CVX_ReLU_MLP + ADMM) ---
    # iter-3: beta 1e-3->1e-4 (lighter group-lasso; 1e-2 over-regularizes to ~chance).
    n_neurons=32, rank=20, beta=1e-4, rho=0.01, gamma_ratio=1.0,
    admm_iters=50, pcg_iters=10, max_feat_dim=None,

    # --- combined source + upweighted calibration fit ---
    # iter-3: cal_balance 1.0->4.0 (more target emphasis; sweep sweet spot 2-4, 8+ hurts low-K).
    source_cap=800,         # max source feature rows in the convex solve (stratified)
    cal_balance=4.0,        # calibration total mass as a fraction of the source rows used
)


def _stratified_subsample(X, y, cap, rng):
    if cap is None or len(X) <= cap:
        return X, y
    classes = np.unique(y)
    per = max(1, cap // len(classes))
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     size=min(per, int((y == c).sum())), replace=False)
                          for c in classes])
    rng.shuffle(idx)
    return X[idx], y[idx]


class ConvexCalibAdapter(BaseAdapter):
    """Source-finetuned MIRepNet + convex head fit on source ∪ upweighted calibration."""

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42, **overrides):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError("ConvexCalibAdapter requires a FoundationBackbone (e.g. mirepnet)")
        super().__init__(backbone, device, seed)
        self.hp = {**HPARAMS, **overrides}
        self._backbone_model: FoundationBackbone | None = None
        self._cld_model = None
        self._feat_mu = self._feat_sigma = None
        self._pca = None
        self._target_R_inv_sqrt: np.ndarray | None = None

    def _align(self, X):
        if not self.hp["use_ea"] or self._target_R_inv_sqrt is None:
            return X
        return euclidean_align(X, self._target_R_inv_sqrt)

    def _source_ft(self, X_src, y_src, n_classes, source_cache):
        h = self.hp
        key = ("convex_calib_sft", self.seed)
        if source_cache is not None and key in source_cache:
            from models.foundations import FoundationWithHead
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            model.load_state_dict(copy.deepcopy(source_cache[key]))
            return model
        model = build_source_finetuned_foundation_model(
            self.backbone, n_classes, X_src, y_src, device=self.device,
            lr_src=h["lr_src"], weight_decay=h["weight_decay"],
            max_epochs_src=h["max_epochs_src"], patience_src=h["patience_src"],
            val_fraction_src=h["val_fraction_src"], batch_size=h["ft_batch_size"],
            seed=self.seed,
        )
        if source_cache is not None:
            source_cache[key] = copy.deepcopy(model.state_dict())
        return model

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        if source_data is None:
            raise ValueError("ConvexCalibAdapter requires source_data")
        self._seed()
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        h = self.hp
        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # 1) optional EA whitening of raw EEG
        if h["use_ea"]:
            if target_unlabeled is None:
                raise ValueError("use_ea=True requires target_unlabeled")
            eps = h["ea_epsilon"]
            self._target_R_inv_sqrt = matrix_sqrt_inv(compute_mean_covariance(target_unlabeled, eps))
            X_src = euclidean_align(X_src, matrix_sqrt_inv(compute_mean_covariance(X_src, eps)))
            target_unlabeled = euclidean_align(target_unlabeled, self._target_R_inv_sqrt)
            if target_labeled is not None:
                target_labeled = (euclidean_align(target_labeled[0], self._target_R_inv_sqrt),
                                  target_labeled[1])

        # 2) source-fine-tune the FM backbone, then freeze (shared w/ baselines, cached)
        model = self._source_ft(X_src, y_src, n_classes, source_cache)
        model.freeze_backbone()
        self._backbone_model = model.backbone.to(self.device)

        # 3) features. normalization stats from large unlabeled target set (stable at low K)
        bs = h["ft_batch_size"]
        X_src_feat = extract_foundation_features(self._backbone_model, X_src, self.device, bs)
        X_src_feat, y_src_sub = _stratified_subsample(X_src_feat, y_src, h["source_cap"], rng)

        if target_unlabeled is not None and len(target_unlabeled) >= 2:
            uk = ("convex_calib_tgt_feats", self.seed)
            if source_cache is not None and uk in source_cache:
                X_unlab = source_cache[uk]
            else:
                X_unlab = extract_foundation_features(self._backbone_model, target_unlabeled, self.device, bs)
                if source_cache is not None:
                    source_cache[uk] = X_unlab
            norm_stats = (X_unlab.mean(0, keepdims=True), X_unlab.std(0, keepdims=True) + 1e-8)
        else:
            norm_stats = None

        # 4) build the combined convex-fit set: source ∪ upweighted calibration
        if target_labeled is not None and len(target_labeled[0]) >= 2:
            X_cal_feat = extract_foundation_features(self._backbone_model, target_labeled[0], self.device, bs)
            y_cal = target_labeled[1]
            cal_mass = h["cal_balance"] * len(X_src_feat)
            reps = max(1, int(round(cal_mass / max(1, len(X_cal_feat)))))
            X_fit = np.concatenate([X_src_feat, np.tile(X_cal_feat, (reps, 1))], axis=0)
            y_fit = np.concatenate([y_src_sub, np.tile(y_cal, reps)], axis=0)
        else:
            X_fit, y_fit = X_src_feat, y_src_sub

        # optional PCA for very high-dim backbones (no-op for MIRepNet's 256-d)
        X_fit, self._pca = maybe_reduce_features(X_fit, h["max_feat_dim"], self.seed)

        # 5) one global convex solve
        self._cld_model, self._feat_mu, self._feat_sigma = fit_cld_head(
            X_fit, y_fit, n_classes, h["n_neurons"], h["rank"], h["beta"], h["rho"],
            h["gamma_ratio"], h["admm_iters"], h["pcg_iters"], self.seed,
            norm_stats=norm_stats if h["max_feat_dim"] is None else None,
        )
        self._fit_time = time.time() - t0
        return self

    def _logits(self, X_feat):
        if self._pca is not None:
            X_feat, _ = maybe_reduce_features(X_feat, self.hp["max_feat_dim"], self.seed, pca=self._pca)
        X_norm = ((X_feat - self._feat_mu) / self._feat_sigma).astype(np.float32)
        return np.array(self._cld_model.stacked_predict(
            jnp.array(X_norm), self._cld_model.theta1, self._cld_model.theta2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat = extract_foundation_features(self._backbone_model, self._align(X), self.device, self.hp["ft_batch_size"])
        return self._logits(feat).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feat = extract_foundation_features(self._backbone_model, self._align(X), self.device, self.hp["ft_batch_size"])
        logits = self._logits(feat)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
