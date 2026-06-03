"""K-adaptive source-anchored convex CLD for SPECIALIST (non-foundation) backbones.

Specialist analogue of ``foundation_sft_kadaptive_anchored_cld.py``. The K-adaptive
Stage-2 objective is identical — an EXPLICIT data-relative quadratic anchor on the
calibration-only solve:

    Stage 2:  min_v  1/2||F(v) - y_cal||^2  +  (a_eff/2)||v - v_anchor||^2  +  beta*grouplasso(v)
    a_eff = a_base * n_ref / n_cal   (strong when calibration is scarce, recedes as K grows)

Only the backbone differs from the foundation variant: instead of a frozen
source-fine-tuned foundation model + foundation features, the backbone is trained
from scratch on pooled source and features are read from the penultimate layer
(exactly as ``anchored_cld.AnchoredCLDAdapter``). The convex Stage-2 helpers
(``_admm_anchored``, ``_build_cld``) are backbone-agnostic and reused as-is.

``anchor_mode="adaptive"`` builds the per-pattern prior a_i ∝ 1/Var_s(v_i^(s)) from a
multi-task per-source-subject convex solve (needs ``source_per_subject`` via the
``source_cache``; falls back to the pooled-source isotropic anchor otherwise).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .cld import pad_features_to_bucket, extract_penultimate_features
from .ea import compute_mean_covariance, euclidean_align, matrix_sqrt_inv
from .anchored_cld import AnchoredCLDAdapter
from .foundation_sft_anchored_cld import _calibration_repeat_count
from .foundation_sft_kadaptive_anchored_cld import _admm_anchored, _build_cld


class KAdaptiveAnchoredCLDAdapter(AnchoredCLDAdapter):
    """Source-trained specialist backbone + K-adaptive explicit-anchor Stage 2.

    Inherits source training, penultimate-feature extraction, Stage 1, and predict from
    ``AnchoredCLDAdapter``; overrides only the Stage-2 solve via ``_fit_stage2``.
    """

    def __init__(self, backbone, device: str = "cpu", seed: int = 42, *,
                 anchor_a_base: float = 2.0, anchor_n_ref: float = 60.0,
                 anchor_mode: str = "adaptive",          # adaptive (per-pattern) | isotropic
                 stage2_data: str = "cal",               # cal | source_cal
                 anchor_var_eps: float = 1e-4,
                 **kwargs):
        # Match the foundation K-adaptive defaults: fixed beta, no union HP-selection
        # (the base's hp_select tunes for the source∪cal UNION, not this explicit anchor).
        kwargs.setdefault("beta", 1e-4)
        kwargs.setdefault("hp_select", False)
        super().__init__(backbone, device, seed, **kwargs)
        self.anchor_a_base = anchor_a_base
        self.anchor_n_ref = anchor_n_ref
        self.anchor_mode = anchor_mode
        self.stage2_data = stage2_data
        self.anchor_var_eps = anchor_var_eps
        self._source_per_subject = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        # per-subject source (for the adaptive per-pattern prior) — from the kwarg or
        # source_cache; if neither is available the adaptive mode falls back to isotropic.
        if source_per_subject is None and source_cache is not None:
            source_per_subject = source_cache.get("source_per_subject")
        self._source_per_subject = source_per_subject
        return super().fit(source_data, target_unlabeled, target_labeled, source_cache)

    # -- per-pattern anchor from a multi-task per-source-subject solve -------------
    def _adaptive_anchor(self, mu, sigma, n_classes, n_neurons, source_cache):
        """Returns (v_bar, a_pattern): mean source head and per-pattern strengths
        a_i ∝ 1/Var_s(v_i), mean-normalized to 1. Cached per target subject."""
        ck = ("kadapt_anchor_spec", self.seed)
        if source_cache is not None and ck in source_cache:
            return source_cache[ck]
        model = self._backbone_model  # source-trained backbone (set by AnchoredCLDAdapter.fit)
        ap = dict(rank=self.rank, beta=self.beta, gamma_ratio=self.gamma_ratio,
                  admm_iters=self.admm_iters, pcg_iters=self.pcg_iters)
        Vs = []
        for X_subj, y_subj in self._source_per_subject:
            f = extract_penultimate_features(model, X_subj, self.device, self.batch_size)
            if self._feat_pca is not None:
                f = self._feat_pca.transform(f).astype(np.float32)
            Xn = ((f - mu) / sigma).astype(np.float32)
            Xn, ys = pad_features_to_bucket(Xn, y_subj.astype(np.int64), 256)
            m = _build_cld(Xn, ys, n_classes, n_neurons, self.beta, self.rho, self.seed)
            _admm_anchored(m, ap, v_anchor=None, anchor_a=0.0)        # cold solve (same gates)
            Vs.append(np.asarray(m.v))
        V = np.stack(Vs, axis=0)                                      # (S, C, 2, d, P)
        v_bar = jnp.asarray(V.mean(0))
        var_i = V.var(0).mean(axis=(0, 1, 2))                         # per-pattern cross-subject var (P,)
        inv = 1.0 / (var_i + self.anchor_var_eps)
        a_pattern = jnp.asarray(inv / inv.mean())                     # mean strength == 1
        out = (v_bar, a_pattern)
        if source_cache is not None:
            source_cache[ck] = out
        return out

    # -- overridden Stage 2: K-adaptive explicit anchor ---------------------------
    def _fit_stage2(self, X_src_feat, y_src, X_calib_feat, y_calib,
                    stage1_model, mu, sigma, n_classes, n_neurons, source_cache=None):
        n_cal = len(X_calib_feat)
        if self.anchor_mode == "adaptive" and self._source_per_subject:
            v_bar, a_pattern = self._adaptive_anchor(mu, sigma, n_classes, n_neurons, source_cache)
        else:
            v_bar, a_pattern = stage1_model.v, 1.0                    # pooled-source head, isotropic

        # data-relative strength: strong when calibration is scarce, recedes as it grows
        scale = self.anchor_a_base * self.anchor_n_ref / max(n_cal, 1)
        a_eff = a_pattern * scale

        if self.stage2_data == "source_cal":
            repeat = _calibration_repeat_count(len(X_src_feat), n_cal, self.target_mass)
            X = np.concatenate([X_src_feat.astype(np.float32),
                                np.repeat(X_calib_feat.astype(np.float32), repeat, axis=0)], axis=0)
            y = np.concatenate([y_src.astype(np.int64),
                                np.repeat(y_calib.astype(np.int64), repeat, axis=0)], axis=0)
        else:  # cal-only (tested best): source enters via the anchor, not by pooling
            X, y = X_calib_feat.astype(np.float32), y_calib.astype(np.int64)

        X_norm = ((X - mu) / sigma).astype(np.float32)
        X_norm, y = pad_features_to_bucket(X_norm, y, 256)
        m = _build_cld(X_norm, y, n_classes, n_neurons, self.beta, self.rho, self.seed)
        _admm_anchored(
            m,
            dict(rank=self.rank, beta=self.beta, gamma_ratio=self.gamma_ratio,
                 admm_iters=self.admm_iters, pcg_iters=self.pcg_iters),
            v_anchor=v_bar, anchor_a=a_eff,
        )
        # Pure target-training time = this Stage-2 solve only (the source-side adaptive
        # anchor build above is cached across K and excluded).
        self._train_time = float(getattr(m, "_solve_time", 0.0))
        return m


class KAdaptiveAnchoredEACLDAdapter(KAdaptiveAnchoredCLDAdapter):
    """EA whitening + source-trained backbone + K-adaptive explicit-anchor Stage 2.

    EA-aligns the source (per-subject when available), target-unlabeled, and calibration
    before the backbone + convex pipeline, then runs the K-adaptive Stage-2 anchor. The
    per-pattern anchor's per-source-subject solve is fed the *aligned* per-subject source
    so the anchor lives in the same whitened space as everything else. Mirrors
    ``anchored_cld.EAAnchoredCLDAdapter`` and ``FoundationSFTKAdaptiveAnchoredEACLDAdapter``.
    """

    # EA whitens the feature space; keep its (would-be) HP cache distinct from non-EA.
    _hp_variant_tag: str = "ea_"

    def __init__(self, backbone, device: str = "cpu", seed: int = 42, *,
                 epsilon: float = 1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        if source_data is None:
            raise ValueError("KAdaptiveAnchoredEACLDAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("KAdaptiveAnchoredEACLDAdapter requires target_unlabeled for EA")

        self._seed()
        X_src, y_src = source_data

        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        if source_per_subject is None and source_cache is not None:
            source_per_subject = source_cache.get("source_per_subject")

        # Align source once (cached across K). Keep aligned per-subject chunks so the
        # K-adaptive per-pattern anchor solves on the same whitened space as Stage 1/2.
        _key = "kadapt_anchored_ea_src_aligned_spec"
        if source_cache is not None and _key in source_cache:
            X_src_aligned, aligned_per_subject = source_cache[_key]
        else:
            if source_per_subject is not None:
                aligned_per_subject = []
                for X_subj, y_subj in source_per_subject:
                    R = compute_mean_covariance(X_subj, self.epsilon)
                    aligned_per_subject.append(
                        (euclidean_align(X_subj, matrix_sqrt_inv(R)), y_subj))
                X_src_aligned = np.concatenate([x for x, _ in aligned_per_subject], axis=0)
            else:
                R_src = compute_mean_covariance(X_src, self.epsilon)
                X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))
                aligned_per_subject = None
            if source_cache is not None:
                source_cache[_key] = (X_src_aligned, aligned_per_subject)

        # Feed the K-adaptive anchor the ALIGNED per-subject source (not the raw kwarg).
        self._source_per_subject = aligned_per_subject

        cal_aligned = None
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            cal_aligned = (euclidean_align(X_cal, self._target_R_inv_sqrt), y_cal)

        # Run the base anchored pipeline directly (skipping KAdaptive.fit, which would reset
        # _source_per_subject to the unaligned data). self._fit_stage2 still resolves to the
        # K-adaptive Stage 2 via the instance.
        return AnchoredCLDAdapter.fit(
            self,
            source_data=(X_src_aligned, y_src),
            target_unlabeled=euclidean_align(target_unlabeled, self._target_R_inv_sqrt),
            target_labeled=cal_aligned,
            source_cache=source_cache,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("KAdaptiveAnchoredEACLDAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("KAdaptiveAnchoredEACLDAdapter not fitted")
        return super().predict_proba(euclidean_align(X, self._target_R_inv_sqrt))
