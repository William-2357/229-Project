"""K-adaptive source-anchored convex adaptation — builds on FoundationSFTAnchoredCLDAdapter.

The base adapter anchors Stage 2 to the source solution *implicitly* (warm-start + a short ADMM on
source∪weighted-calibration). This subclass replaces that with an EXPLICIT quadratic anchor on the
calibration-only solve:

    Stage 2:  min_v  1/2||F(v) - y_cal||^2  +  (a_eff/2)||v - v_src||^2  +  beta*grouplasso(v)

with a DATA-RELATIVE anchor strength

    a_eff = a_base * n_ref / n_cal

so the source prior is STRONG when calibration is scarce (low K — fills the underdetermined null
space) and RECEDES as calibration grows (high K — data dominates). This removes the sensitivity of a
fixed anchor strength (a single fixed `a` either craters at low K or flattens at high K).

`anchor_mode="adaptive"` makes it per-pattern (Mahalanobis-spirit): a_i ∝ 1/Var_s(v_i^(s)) from a
multi-task per-source-subject solve — cross-subject-conserved neurons are anchored hard, variable
ones are left free to fit the target. Requires `source_per_subject`; falls back to isotropic otherwise.

Tested (auto branch, NeuroGPT full-dim): a_base*n_ref=120, cal-only, adaptive -> 0.6777, which BEATS
the source∪cal union (0.672 frozen / 0.676 LoRA) and the sft_lora baseline (0.656). Fixed-a anchoring
was 0.654 (+0.024 from the data-relative fix). Wins where source is less task-aligned; on MIRepNet
(MI-pretrained, source spans the task) the union still wins -> backbone-dependent.
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP
from jaxcld.optimizers.pcg import pcg
from jaxcld.preconditioner.nystrom import Nys_Precond
from jaxcld.utils.proximal_utils import batch_proxl2_tensor
from ._jaxcld_cpu_linalg import rand_nys_appx_cpu as rand_nys_appx

from .cld import pad_features_to_bucket
from .foundation_cld import extract_foundation_features
from .ea import compute_mean_covariance, euclidean_align, matrix_sqrt_inv
from .foundation_sft_anchored_cld import (
    FoundationSFTAnchoredCLDAdapter,
    _calibration_repeat_count,
)


def _prox_group_l2(z, thresh):
    """Group-L2 prox over the feature axis (per class, per neuron-column).
    z: (n_classes, d, P_S); thresh: scalar or (P_S,). Generalizes jaxcld.proxl2_tensor to a
    per-column threshold so the anchor strength can be per-pattern."""
    norms = jnp.linalg.norm(z, axis=1, keepdims=True)                 # (C,1,P)
    return z * jnp.maximum(0.0, 1.0 - thresh / jnp.maximum(norms, 1e-12))


def _admm_anchored(model, admm_params, v_anchor, anchor_a, u_init=None, v_init=None):
    """Warm-startable jaxcld ADMM with an explicit quadratic anchor (a/2)||v - v_anchor||^2.

    Mirrors foundation_sft_anchored_cld._admm_warm; the only change is the v-update, which blends
    the ADMM point toward the anchor and shrinks with the (rho+a)-adjusted prox — a closed form:
        q = (rho*(u+lam) + a*v_anchor)/(rho+a);   v = prox_{beta/(rho+a)}(q).
    `anchor_a` is a scalar (isotropic) or a (P_S,) array (per-pattern). a==0 recovers the stock warm
    ADMM. Slacks/duals (s, nu) are re-zeroed (their shape is sample-count dependent)."""
    rank = admm_params["rank"]; beta = admm_params["beta"]; gamma_ratio = admm_params["gamma_ratio"]
    admm_iters = admm_params["admm_iters"]; pcg_iters = admm_params["pcg_iters"]
    n, d = model.X.shape; rho = model.rho; C, P = model.n_classes, model.P_S
    Y = jax.nn.one_hot(model.y, C)
    wshape = (C, 2, d, P); sshape = (C, 2, n, P)

    u = jnp.asarray(u_init) if u_init is not None else jnp.zeros(wshape)
    v = jnp.asarray(v_init) if v_init is not None else jnp.zeros(wshape)
    lam = jnp.zeros(wshape); s = jnp.zeros(sshape); nu = jnp.zeros(sshape)

    a = jnp.asarray(anchor_a, dtype=jnp.float32)                      # scalar or (P,)
    va = jnp.zeros(wshape) if (v_anchor is None) else jnp.asarray(v_anchor)
    if v_anchor is None:
        a = jnp.zeros((P,), dtype=jnp.float32)
    thresh = beta / (rho + a)                                         # per-column prox threshold

    # Pure-solve wall clock (Nyström setup + all ADMM iters) -> model._solve_time,
    # read into the adapter's train_time. First call per shape includes the XLA
    # compile (warm-repeat mean is the compile-free figure; cf. fit_time_warm).
    _t_solve_start = time.perf_counter()
    U, S_nys, model.seed = rand_nys_appx(model, rank, model.seed)
    Mnys = Nys_Precond(U, S_nys, d, model.rho, model.P_S)
    b_1 = model.batch_rmatvec_F(Y.T) / model.rho

    for _ in range(admm_iters):
        b = b_1 + v - lam + model.batch_rmatvec_G(s - nu)
        u, _, _ = pcg(b, model, Mnys, pcg_iters)
        q0 = (rho * (u[:, 0, :] + lam[:, 0, :]) + a * va[:, 0, :]) / (rho + a)
        q1 = (rho * (u[:, 1, :] + lam[:, 1, :]) + a * va[:, 1, :]) / (rho + a)
        v = v.at[:, 0, :].set(_prox_group_l2(q0, thresh))
        v = v.at[:, 1, :].set(_prox_group_l2(q1, thresh))
        Gu = model.batch_matvec_G(u)
        s = jax.nn.relu(Gu + nu)
        lam = lam + (u - v) * gamma_ratio
        nu = nu + (Gu - s) * gamma_ratio

    model.u, model.v, model.s, model.lam, model.nu = u, v, s, lam, nu
    W1, w2 = model.get_ncvx_weights(v)
    model.theta1, model.theta2 = W1, w2
    jax.block_until_ready([model.theta1, model.theta2])
    model._solve_time = time.perf_counter() - _t_solve_start


def _build_cld(X_norm, y, n_classes, n_neurons, beta, rho, seed):
    """CVX_ReLU_MLP with hyperplanes from `seed` (data-independent -> shared across stages)."""
    m = CVX_ReLU_MLP(X=jnp.asarray(X_norm), y=jnp.asarray(y.astype(np.int32)),
                     n_classes=n_classes, P_S=n_neurons, beta=beta, rho=rho,
                     seed=jax.random.PRNGKey(seed))
    m.init_model()
    return m


class FoundationSFTKAdaptiveAnchoredCLDAdapter(FoundationSFTAnchoredCLDAdapter):
    """SFT backbone + Stage-1 source convex head + K-adaptive explicit-anchor Stage 2.

    Inherits SFT, Stage-1 (source cold ADMM), feature extraction (+PCA), and predict from the base;
    overrides only the Stage-2 solve. Defaults reproduce the tested NeuroGPT-best config.
    """

    def __init__(self, backbone, device: str = "cpu", seed: int = 42, *,
                 anchor_a_base: float = float(os.environ.get("KADAPT_A_BASE", "2.0")),
                 anchor_n_ref: float = 60.0,
                 # env overrides let us A/B anchor variants on Modal without code churn
                 anchor_mode: str = os.environ.get("KADAPT_ANCHOR_MODE", "adaptive"),  # adaptive | isotropic
                 stage2_data: str = os.environ.get("KADAPT_STAGE2", "cal"),            # cal | source_cal
                 anchor_var_eps: float = 1e-4,
                 max_feat_dim: int | None = None,         # None = full dim (tested best); 256 = PCA
                 **kwargs):
        # tested-best base config for this method: fixed beta=1e-4 and NO union HP-selection
        # (the base's hp_select tunes beta/target_mass for the source∪cal UNION, not this anchor).
        kwargs.setdefault("beta", 1e-4)
        kwargs.setdefault("hp_select", False)
        # KADAPT_SFT_EPOCHS=0 => skip source fine-tuning, use the RAW FROZEN backbone features
        # (matches the auto repo's "frozen" config; SFT can overfit source and hurt cal-only transfer).
        if "KADAPT_SFT_EPOCHS" in os.environ:
            kwargs.setdefault("max_epochs_src", int(os.environ["KADAPT_SFT_EPOCHS"]))
        super().__init__(backbone, device, seed, max_feat_dim=max_feat_dim, **kwargs)
        self.anchor_a_base = anchor_a_base
        self.anchor_n_ref = anchor_n_ref
        self.anchor_mode = anchor_mode
        self.stage2_data = stage2_data
        self.anchor_var_eps = anchor_var_eps
        self._source_per_subject = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        # per-subject source (for the adaptive per-pattern prior) — from the kwarg or source_cache;
        # if neither is available the adaptive mode falls back to isotropic. Rest is the base pipeline.
        if source_per_subject is None and source_cache is not None:
            source_per_subject = source_cache.get("source_per_subject")
        self._source_per_subject = source_per_subject
        return super().fit(source_data, target_unlabeled, target_labeled, source_cache)

    # -- per-pattern anchor from a multi-task per-source-subject solve -------------
    def _adaptive_anchor(self, mu, sigma, n_classes, n_neurons, source_cache):
        """Returns (v_bar, a_pattern): mean source head and per-pattern strengths a_i ∝ 1/Var_s(v_i),
        mean-normalized to 1. Cached per target subject (depends only on source)."""
        ck = ("kadapt_anchor", self.seed)
        if source_cache is not None and ck in source_cache:
            return source_cache[ck]
        bk = self._backbone_model.to(self.device)
        ap = dict(rank=self.rank, beta=self.beta, gamma_ratio=self.gamma_ratio,
                  admm_iters=self.admm_iters, pcg_iters=self.pcg_iters)
        Vs = []
        for X_subj, y_subj in self._source_per_subject:
            f = extract_foundation_features(bk, X_subj, self.device, self.batch_size)
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
        # cold-start + explicit anchor, full admm_iters — as tested (the anchor, not warm-starting,
        # carries the source information; the convex solve converges to the same anchored optimum).
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


class FoundationSFTKAdaptiveAnchoredEACLDAdapter(FoundationSFTKAdaptiveAnchoredCLDAdapter):
    """EA whitening + SFT backbone + K-adaptive explicit-anchor Stage 2.

    EA-aligns the source (per-subject when available), target-unlabeled, and calibration before the
    convex pipeline, then runs the exact K-adaptive Stage-2 anchor of the parent. The per-pattern
    anchor's multi-task per-source-subject solve is fed the *aligned* per-subject source so the
    anchor lives in the same whitened feature space as everything else.

    Mirrors ``FoundationSFTAnchoredEACLDAdapter`` but for the K-adaptive Stage 2.
    """

    # EA whitens the feature space; keep its (would-be) HP cache distinct from the non-EA variant.
    _hp_variant_tag: str = "ea_"

    def __init__(self, backbone, device: str = "cpu", seed: int = 42, *,
                 epsilon: float = 1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        if source_data is None:
            raise ValueError("FoundationSFTKAdaptiveAnchoredEACLDAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError(
                "FoundationSFTKAdaptiveAnchoredEACLDAdapter requires target_unlabeled for EA")

        self._seed()
        X_src, y_src = source_data

        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        if source_per_subject is None and source_cache is not None:
            source_per_subject = source_cache.get("source_per_subject")

        # Align the source once (cached across K). Keep the aligned *per-subject* chunks so the
        # K-adaptive per-pattern anchor solves on the same whitened space as Stage 1/2.
        _ea_src_key = "sft_kadapt_anchored_ea_src_aligned"
        if source_cache is not None and _ea_src_key in source_cache:
            X_src_aligned, aligned_per_subject = source_cache[_ea_src_key]
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
                source_cache[_ea_src_key] = (X_src_aligned, aligned_per_subject)

        # Feed the K-adaptive anchor the ALIGNED per-subject source (not the raw kwarg).
        self._source_per_subject = aligned_per_subject

        cal_aligned = None
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            cal_aligned = (euclidean_align(X_cal, self._target_R_inv_sqrt), y_cal)

        # Run the base anchored pipeline directly (skipping the parent's fit, which would reset
        # _source_per_subject to the unaligned data). self._fit_stage2 still resolves to the
        # K-adaptive Stage 2 via the instance.
        return FoundationSFTAnchoredCLDAdapter.fit(
            self,
            source_data=(X_src_aligned, y_src),
            target_unlabeled=euclidean_align(target_unlabeled, self._target_R_inv_sqrt),
            target_labeled=cal_aligned,
            source_cache=source_cache,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSFTKAdaptiveAnchoredEACLDAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSFTKAdaptiveAnchoredEACLDAdapter not fitted")
        return super().predict_proba(euclidean_align(X, self._target_R_inv_sqrt))
