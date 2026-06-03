"""Source-anchored 2-stage convex adaptation for SFT foundation backbones.

Pipeline:
  1. SFT stage  — fine-tune backbone + head on pooled source subjects (same as
                  foundation_sft_cld; checkpoint-cached across K values).
  2. Stage 1    — cold ADMM on source features → CVX_ReLU_MLP; primal variables
                  (u, v, lam) saved for warm-starting.
  3. Stage 2    — when K > 0: rebuild CVX_ReLU_MLP on source + weighted
                  calibration features (same random seed → same hyperplanes),
                  warm-start ADMM from Stage 1 primal, reset slacks/duals.
                  When K = 0: Stage 1 model is the predictor.

The source-anchored objective (Stage 2) keeps the convex head from collapsing
on small K by anchoring to the source solution rather than treating source data
as merely an ADMM warm start.

Reference: reve_kmin_convexnn_v3.ipynb — adapted for BCICIV2a LOSO and
           foundation backbones (CBraMod / LaBraM / MIRepNet / NeuroGPT).
"""

from __future__ import annotations

import copy
import hashlib
import os
import time

import jax
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_compilation_cache_dir", "/root/.cache/jax_xla")
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP
from jaxcld.optimizers.pcg import pcg
from jaxcld.preconditioner.nystrom import Nys_Precond
from jaxcld.utils.proximal_utils import batch_proxl2_tensor

# CPU-pinned Nyström build (qr/cholesky/solve/svd) — avoids cuSolver INTERNAL
# crashes on Modal GPUs. Same numerics as jaxcld's rand_nys_appx.
from ._jaxcld_cpu_linalg import rand_nys_appx_cpu as rand_nys_appx

from .base import BaseAdapter
from .cld import maybe_reduce_features, pad_features_to_bucket
from .ea import compute_mean_covariance, euclidean_align, matrix_sqrt_inv
from .foundation_cld import extract_foundation_features
from .foundation_source_finetune import build_source_finetuned_foundation_model
from models.foundations import FoundationBackbone, FoundationWithHead


# ---------------------------------------------------------------------------
# Warm-startable ADMM
# ---------------------------------------------------------------------------

# Set CLD_TIMING=1 to print a per-call compile-vs-solve breakdown for the ADMM
# solver. Iteration 0 pays the XLA compile (the Python loop calls jitted ops),
# iterations 1..N are warm, so the gap isolates compile from steady-state solve.
_CLD_TIMING = os.environ.get("CLD_TIMING", "") not in ("", "0", "false", "False")


def _admm_warm(
    model: CVX_ReLU_MLP,
    admm_params: dict,
    u_init=None,
    v_init=None,
    lam_init=None,
    label: str = "admm",
):
    """ADMM with optional primal warm-start on (u, v, lam).

    Standard jaxcld admm() does not save lam, so Stage 2 warm-starting is
    impossible with the library version. This inline copy adds:
      - optional (u_init, v_init, lam_init) for the primal/dual variables
      - always saves lam on the model so Stage 2 can read it
    Slack/dual variables (s, nu) are always re-zeroed because their shape
    depends on n_samples, which changes between stages.
    """
    rank = admm_params["rank"]
    beta = admm_params["beta"]
    gamma_ratio = admm_params["gamma_ratio"]
    admm_iters = admm_params["admm_iters"]
    pcg_iters = admm_params["pcg_iters"]

    n, d = model.X.shape
    Y = jax.nn.one_hot(model.y, model.n_classes)

    weight_shape = (model.n_classes, 2, d, model.P_S)
    sample_shape = (model.n_classes, 2, n, model.P_S)

    u = jnp.asarray(u_init) if u_init is not None else jnp.zeros(weight_shape)
    v = jnp.asarray(v_init) if v_init is not None else jnp.zeros(weight_shape)
    lam = jnp.asarray(lam_init) if lam_init is not None else jnp.zeros(weight_shape)
    s = jnp.zeros(sample_shape)
    nu = jnp.zeros(sample_shape)

    if _CLD_TIMING:
        _t_setup = time.perf_counter()
    U, S_nys, model.seed = rand_nys_appx(model, rank, model.seed)
    Mnys = Nys_Precond(U, S_nys, d, model.rho, model.P_S)
    b_1 = model.batch_rmatvec_F(Y.T) / model.rho
    if _CLD_TIMING:
        jax.block_until_ready((U, S_nys, b_1))
        _setup_s = time.perf_counter() - _t_setup
        _iter_times = []

    for _ in range(admm_iters):
        if _CLD_TIMING:
            _t_it = time.perf_counter()
        b = b_1 + v - lam + model.batch_rmatvec_G(s - nu)
        u, _, _ = pcg(b, model, Mnys, pcg_iters)
        v = v.at[:, 0, :].set(
            batch_proxl2_tensor(u[:, 0, :] + lam[:, 0, :], beta=beta, gamma=1.0 / model.rho)
        )
        v = v.at[:, 1, :].set(
            batch_proxl2_tensor(u[:, 1, :] + lam[:, 1, :], beta=beta, gamma=1.0 / model.rho)
        )
        Gu = model.batch_matvec_G(u)
        s = jax.nn.relu(Gu + nu)
        lam = lam + (u - v) * gamma_ratio
        nu = nu + (Gu - s) * gamma_ratio
        if _CLD_TIMING:
            jax.block_until_ready((u, v, s, lam, nu))
            _iter_times.append(time.perf_counter() - _t_it)

    if _CLD_TIMING:
        _warm = _iter_times[1:] if len(_iter_times) > 1 else _iter_times
        _warm_mean = sum(_warm) / len(_warm)
        _compile_est = _iter_times[0] - _warm_mean
        print(
            f"[cld-timing] {label} backend={jax.default_backend()} "
            f"n={n} d={d} P_S={model.P_S} admm_iters={admm_iters} pcg_iters={pcg_iters} "
            f"setup(nystrom+b1)={_setup_s:.3f}s "
            f"iter0(compile+exec)={_iter_times[0]:.3f}s "
            f"warm_iter_mean={_warm_mean:.4f}s "
            f"solve(warm*iters)={_warm_mean * admm_iters:.3f}s "
            f"est_compile={_compile_est:.3f}s "
            f"total={_setup_s + sum(_iter_times):.3f}s",
            flush=True,
        )

    model.u = u
    model.v = v
    model.s = s
    model.lam = lam
    model.nu = nu
    W1, w2 = model.get_ncvx_weights(v)
    model.theta1 = W1
    model.theta2 = w2


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _calibration_repeat_count(n_src: int, n_calib: int, target_mass: float = 0.35) -> int:
    """Number of times to row-repeat calibration trials to approximate target_mass weight."""
    target_mass = float(np.clip(target_mass, 1e-3, 0.95))
    odds = target_mass / (1.0 - target_mass)
    return max(1, int(round(odds * n_src / n_calib)))


def fit_stage1_source(
    X_feat: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_neurons: int,
    rank: int,
    beta: float,
    rho: float,
    gamma_ratio: float,
    admm_iters: int,
    pcg_iters: int,
    seed: int,
    pad_bucket: int | None = 256,
) -> tuple[CVX_ReLU_MLP, np.ndarray, np.ndarray]:
    """Stage 1: cold ADMM on source features. Saves (u, v, lam) on the model."""
    mu = X_feat.mean(axis=0, keepdims=True)
    sigma = X_feat.std(axis=0, keepdims=True) + 1e-8
    X_norm = ((X_feat - mu) / sigma).astype(np.float32)
    X_norm, y = pad_features_to_bucket(X_norm, y, pad_bucket)

    key = jax.random.PRNGKey(seed)
    m = CVX_ReLU_MLP(
        X=jnp.asarray(X_norm), y=jnp.asarray(y.astype(np.int32)),
        n_classes=n_classes, P_S=n_neurons,
        beta=beta, rho=rho, seed=key,
    )
    m.init_model()
    _admm_warm(m, dict(rank=rank, beta=beta, gamma_ratio=gamma_ratio,
                       admm_iters=admm_iters, pcg_iters=pcg_iters),
               label="stage1_source")
    return m, mu, sigma


def fit_stage2_anchored(
    X_src_feat: np.ndarray,
    y_src: np.ndarray,
    X_calib_feat: np.ndarray,
    y_calib: np.ndarray,
    stage1_model: CVX_ReLU_MLP,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_classes: int,
    n_neurons: int,
    rank: int,
    beta: float,
    rho: float,
    gamma_ratio: float,
    admm_iters: int,
    pcg_iters: int,
    seed: int,
    target_mass: float = 0.35,
    pad_bucket: int | None = 256,
) -> CVX_ReLU_MLP:
    """Stage 2: warm ADMM on source + weighted calibration (source-anchored).

    Calibration trials are row-repeated to approximate target_mass weight in
    the squared loss. The Stage 1 scaler (mu, sigma) is reused so the
    warm-started primal variables (u, v) remain valid in the same feature space.
    lam is reset (Stage 1 dual is tied to Stage 1 stationarity equations).
    """
    repeat = _calibration_repeat_count(len(X_src_feat), len(X_calib_feat), target_mass)
    X_calib_rep = np.repeat(X_calib_feat.astype(np.float32), repeat, axis=0)
    y_calib_rep = np.repeat(y_calib.astype(np.int64), repeat, axis=0)

    X_aug = np.concatenate([X_src_feat.astype(np.float32), X_calib_rep], axis=0)
    y_aug = np.concatenate([y_src.astype(np.int64), y_calib_rep], axis=0)
    X_aug_norm = ((X_aug - mu) / sigma).astype(np.float32)

    # Pad to a row count that depends ONLY on (n_src, target_mass) — both fixed for
    # a sweep — so the jitted ADMM/PCG solver compiles ONCE across K. X_aug = source
    # + repeated calibration, and the repeat count varies with n_calib (K), so
    # pad_features_to_bucket (next multiple) lets the size cross 256-buckets and
    # retrace per call. Since repeat*n_calib ≈ odds*n_src (≈K-invariant), n_src*(1+odds)
    # + one bucket of slack upper-bounds X_aug across K; appended zero rows are
    # solution-invariant. Fallback rounds up if a rare large-K calib exceeds the bound.
    if pad_bucket:
        n_now = X_aug_norm.shape[0]
        tm = float(np.clip(target_mass, 1e-3, 0.95))
        odds = tm / (1.0 - tm)
        n_expected = len(X_src_feat) + int(np.ceil(odds * len(X_src_feat)))
        n_fixed = int(np.ceil(n_expected / pad_bucket)) * pad_bucket + pad_bucket
        n_target = n_fixed if n_now <= n_fixed else int(np.ceil(n_now / pad_bucket)) * pad_bucket
        if n_target > n_now:
            pad = n_target - n_now
            X_aug_norm = np.concatenate(
                [X_aug_norm, np.zeros((pad, X_aug_norm.shape[1]), dtype=X_aug_norm.dtype)], axis=0)
            y_aug = np.concatenate([y_aug, np.zeros(pad, dtype=y_aug.dtype)], axis=0)

    # Same seed → same random hyperplanes → warm-started weights are in the same basis.
    key = jax.random.PRNGKey(seed)
    m = CVX_ReLU_MLP(
        X=jnp.asarray(X_aug_norm), y=jnp.asarray(y_aug.astype(np.int32)),
        n_classes=n_classes, P_S=n_neurons,
        beta=beta, rho=rho, seed=key,
    )
    m.init_model()
    _admm_warm(
        m,
        dict(rank=rank, beta=beta, gamma_ratio=gamma_ratio,
             admm_iters=admm_iters, pcg_iters=pcg_iters),
        u_init=stage1_model.u,
        v_init=stage1_model.v,
        lam_init=None,  # reset: Stage 1 dual is stale for the augmented problem
        label="stage2_anchored",
    )
    return m


def _predict_from_cld(model: CVX_ReLU_MLP, mu: np.ndarray, sigma: np.ndarray,
                      X_feat: np.ndarray) -> np.ndarray:
    X_norm = ((X_feat - mu) / sigma).astype(np.float32)
    return np.array(model.stacked_predict(jnp.asarray(X_norm), model.theta1, model.theta2))


# ---------------------------------------------------------------------------
# Backbone-agnostic hyperparameter selection (shared by foundation + specialist)
# ---------------------------------------------------------------------------

class _AnchoredHPSelectMixin:
    """Leak-free (beta, target_mass) selection for source-anchored 2-stage CLD.

    Operates only on source feature matrices and the ADMM hyperparameters held on
    ``self`` (rank/rho/gamma_ratio/admm_iters/admm_iters_stage2/pcg_iters/grids/
    hp_val_k/seed) plus a per-backbone disk cache path, so it is independent of how
    the features were produced. Shared by the foundation (SFT) and specialist
    anchored-CLD adapters; subclasses set ``_hp_variant_tag`` to keep EA and non-EA
    selections in distinct cache files.
    """

    # Tag in the per-backbone HP cache filename; EA subclasses tune separately.
    _hp_variant_tag: str = ""

    def _select_hparams(
        self, X_src_feat: np.ndarray, y_src: np.ndarray,
        n_classes: int, n_neurons: int,
    ) -> tuple[float, float]:
        """Pick (beta, target_mass) on a source-internal validation split.

        A stratified slice of the source acts as a pseudo-target: its first
        ``hp_val_k`` trials/class are pseudo-calibration, the remainder
        pseudo-eval. For each (beta, target_mass) we run the same source-anchored
        2-stage solve and score pseudo-eval accuracy; the best combo wins. Uses
        source data only (never the held-out subject), so it is leak-free.
        Mirrors the notebook's beta x target_mass grid (cells 20-21) but selected
        per fold on validation rather than reported on the test subjects.
        """
        rng = np.random.RandomState(self.seed)
        ps_idx, pt_idx = [], []
        for c in range(n_classes):
            ci = np.where(y_src == c)[0]
            if len(ci) < 2 * self.hp_val_k + 1:
                return self.beta, self.target_mass  # too few to validate; keep defaults
            rng.shuffle(ci)
            n_pt = max(2 * self.hp_val_k, int(0.2 * len(ci)))
            pt_idx.extend(ci[:n_pt].tolist())
            ps_idx.extend(ci[n_pt:].tolist())
        X_ps, y_ps = X_src_feat[np.asarray(ps_idx)], y_src[np.asarray(ps_idx)]
        X_pt, y_pt = X_src_feat[np.asarray(pt_idx)], y_src[np.asarray(pt_idx)]

        cal_idx, ev_idx = [], []
        for c in range(n_classes):
            ci = np.where(y_pt == c)[0]
            cal_idx.extend(ci[:self.hp_val_k].tolist())
            ev_idx.extend(ci[self.hp_val_k:].tolist())
        X_cal, y_cal = X_pt[np.asarray(cal_idx)], y_pt[np.asarray(cal_idx)]
        X_ev, y_ev = X_pt[np.asarray(ev_idx)], y_pt[np.asarray(ev_idx)]
        if len(X_ev) == 0 or len(X_cal) == 0:
            return self.beta, self.target_mass

        best, best_acc = (self.beta, self.target_mass), -1.0
        for beta in self.beta_grid:
            s1, mu, sigma = fit_stage1_source(
                X_ps, y_ps, n_classes=n_classes, n_neurons=n_neurons,
                rank=self.rank, beta=beta, rho=self.rho,
                gamma_ratio=self.gamma_ratio, admm_iters=self.admm_iters,
                pcg_iters=self.pcg_iters, seed=self.seed,
            )
            for tm in self.target_mass_grid:
                m = fit_stage2_anchored(
                    X_ps, y_ps, X_cal, y_cal, s1, mu, sigma,
                    n_classes=n_classes, n_neurons=n_neurons,
                    rank=self.rank, beta=beta, rho=self.rho,
                    gamma_ratio=self.gamma_ratio, admm_iters=self.admm_iters_stage2,
                    pcg_iters=self.pcg_iters, seed=self.seed, target_mass=tm,
                )
                pred = _predict_from_cld(m, mu, sigma, X_ev).argmax(axis=1)
                acc = float((pred == y_ev).mean())
                if acc > best_acc:
                    best_acc, best = acc, (beta, tm)
        print(f"[anchored-cld] HP select: beta={best[0]:g} target_mass={best[1]:.2f} "
              f"(val_acc={best_acc:.3f})", flush=True)
        return best

    def _hp_cache_path(self) -> str | None:
        """Disk path for the tune-once-per-backbone HPs (Modal volume).

        Returns None off-Modal (no /data), in which case selection falls back to
        the per-fold in-process path. The key is per backbone + seed, so a single
        job selects (beta, target_mass) once and all other folds/methods reuse it.
        """
        if not os.path.isdir("/data"):
            return None
        backbone_name = self.backbone.__class__.__name__.lower()
        return os.path.join(
            "/data/sft_checkpoints",
            f"{backbone_name}_anchored_{self._hp_variant_tag}hp_seed{self.seed}.json",
        )

    def _resolve_hparams(
        self, X_src_feat: np.ndarray, y_src: np.ndarray,
        n_classes: int, n_neurons: int,
    ) -> tuple[float, float]:
        """Tune once per backbone: reuse disk-persisted (beta, target_mass) if a
        prior job already selected it; otherwise select now and persist for the
        rest of the run. Selection itself (`_select_hparams`) is leak-free."""
        import json
        path = self._hp_cache_path()
        if path is not None and os.path.exists(path):
            try:
                hp = json.loads(open(path).read())
                print(f"[anchored-cld] HP reuse (per-backbone): beta={hp['beta']:g} "
                      f"target_mass={hp['target_mass']:.2f}", flush=True)
                return float(hp["beta"]), float(hp["target_mass"])
            except Exception:
                pass
        beta, tm = self._select_hparams(X_src_feat, y_src, n_classes, n_neurons)
        if path is not None:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").write(json.dumps({"beta": beta, "target_mass": tm}))
                import __main__
                if hasattr(__main__, "data_volume"):
                    __main__.data_volume.commit()
            except Exception:
                pass
        return beta, tm


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FoundationSFTAnchoredCLDAdapter(_AnchoredHPSelectMixin, BaseAdapter):
    """SFT backbone + source-anchored 2-stage convex CLD head.

    Stage 1 (always): fit CVX_ReLU_MLP on source features via cold ADMM.
    Stage 2 (K > 0):  refit on source + weighted calibration, warm-starting
                      Stage 1 primal variables (u, v).
    K = 0:            Stage 1 model used directly (zero-shot).
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        # SFT params
        lr_src: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_src: int = 200,
        patience_src: int = 25,
        val_fraction_src: float = 0.1,
        batch_size: int = 32,
        # CLD / ADMM params
        rank: int = 20,
        beta: float = 1e-3,
        rho: float = 0.01,
        gamma_ratio: float = 1.0,
        admm_iters: int = 50,
        pcg_iters: int = 10,
        n_neurons: int | None = None,
        # Stage-2-specific
        admm_iters_stage2: int = 10,
        target_mass: float = 0.35,
        max_feat_dim: int | None = None,  # None = no PCA; CLD runs on full features
        # Stage-2 HP grid — validation-selected per fold (mirrors notebook grid).
        hp_select: bool = True,
        beta_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3),
        target_mass_grid: tuple[float, ...] = (0.15, 0.35, 0.55),
        hp_val_k: int = 4,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationSFTAnchoredCLDAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.batch_size = batch_size
        self.rank = rank
        self.beta = beta
        self.rho = rho
        self.gamma_ratio = gamma_ratio
        self.admm_iters = admm_iters
        self.pcg_iters = pcg_iters
        self.n_neurons = n_neurons
        self.admm_iters_stage2 = admm_iters_stage2
        self.target_mass = target_mass
        self.max_feat_dim = max_feat_dim
        self.hp_select = hp_select
        self.beta_grid = beta_grid
        self.target_mass_grid = target_mass_grid
        self.hp_val_k = hp_val_k

        self._backbone_model: FoundationBackbone | None = None
        self._cld_model: CVX_ReLU_MLP | None = None
        self._feat_mu: np.ndarray | None = None
        self._feat_sigma: np.ndarray | None = None
        self._feat_pca = None

    def _sft_kwargs(self) -> dict:
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
        self,
        source_data,
        target_unlabeled=None,
        target_labeled=None,
        source_cache: dict | None = None,
    ) -> "FoundationSFTAnchoredCLDAdapter":
        if source_data is None:
            raise ValueError("FoundationSFTAnchoredCLDAdapter requires source_data")

        self._seed()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))
        n_neurons = self.n_neurons or (10 if n_classes == 2 else 32)

        # ---- Frozen SFT backbone — checkpoint-cached on disk, in-memory across K.
        # One-time shared source-side cost, built BEFORE the fit_time clock.
        backbone_name = self.backbone.__class__.__name__.lower()
        checkpoint_dir = "/data/sft_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        volume_needs_commit = False
        _frozen_key = "sft_anchored_frozen_backbone"
        if source_cache is not None and _frozen_key in source_cache:
            self._backbone_model = source_cache[_frozen_key]
        else:
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            src_hash = hashlib.md5(X_src.tobytes()[:50000] + y_src.tobytes()).hexdigest()[:8]
            # Hash the pretrained backbone weights so an updated checkpoint busts the SFT
            # cache (SFT starts from these weights; without this a new .pth silently reuses
            # the stale source-fine-tuned checkpoint).
            bb_hash = hashlib.md5(b"".join(
                v.detach().cpu().numpy().tobytes()[:4096]
                for v in self.backbone.state_dict().values()
            )).hexdigest()[:6]
            lr_tag = f"lr{self.lr_src:.0e}".replace("-", "n")
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{backbone_name}_seed{self.seed}_src_{src_hash}_bb{bb_hash}_{lr_tag}_ep{self.max_epochs_src}_sft.pt",
            )
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            else:
                model = build_source_finetuned_foundation_model(
                    self.backbone, n_classes, X_src, y_src, **self._sft_kwargs()
                )
                torch.save(model.state_dict(), checkpoint_path)
                volume_needs_commit = True
            model.freeze_backbone()
            self._backbone_model = model.backbone
            if source_cache is not None:
                source_cache[_frozen_key] = self._backbone_model

        if volume_needs_commit:
            try:
                import __main__
                if hasattr(__main__, "data_volume"):
                    __main__.data_volume.commit()
            except Exception:
                pass

        # ---- Source features (+ optional PCA), cached across K --------------
        backbone = self._backbone_model.to(self.device)
        _src_feat_key = "sft_anchored_cld_src_feats"
        if source_cache is not None and _src_feat_key in source_cache:
            X_src_feat, self._feat_pca = source_cache[_src_feat_key]
        else:
            X_src_feat = extract_foundation_features(backbone, X_src, self.device, self.batch_size)
            X_src_feat, self._feat_pca = maybe_reduce_features(X_src_feat, self.max_feat_dim, self.seed)
            if source_cache is not None:
                source_cache[_src_feat_key] = (X_src_feat, self._feat_pca)

        # ---- Stage-2 HP selection + Stage 1, cached across K ----------------
        # (beta, target_mass) are picked once per fold on a leak-free source
        # validation split; Stage 1 is then fit with the chosen beta. Both are
        # one-time source-side costs, so they sit BEFORE the fit_time clock.
        has_calib = target_labeled is not None and len(target_labeled[0]) >= n_classes
        _stage1_key = "sft_anchored_stage1"
        if source_cache is not None and _stage1_key in source_cache:
            self.beta, self.target_mass, stage1_model, mu, sigma = source_cache[_stage1_key]
        else:
            if self.hp_select and has_calib:
                self.beta, self.target_mass = self._resolve_hparams(
                    X_src_feat, y_src, n_classes, n_neurons)
            stage1_model, mu, sigma = fit_stage1_source(
                X_src_feat, y_src, n_classes=n_classes, n_neurons=n_neurons,
                rank=self.rank, beta=self.beta, rho=self.rho,
                gamma_ratio=self.gamma_ratio, admm_iters=self.admm_iters,
                pcg_iters=self.pcg_iters, seed=self.seed,
            )
            if source_cache is not None:
                source_cache[_stage1_key] = (self.beta, self.target_mass, stage1_model, mu, sigma)
        self._feat_mu = mu
        self._feat_sigma = sigma

        # ---- fit_time covers per-K target adaptation only (Stage 2) ---------
        t0 = time.time()

        # ---- Stage 2: warm ADMM on source + calibration (K > 0) ------------
        if target_labeled is not None and len(target_labeled[0]) >= n_classes:
            X_calib, y_calib = target_labeled
            X_calib_feat = extract_foundation_features(backbone, X_calib, self.device, self.batch_size)
            if self._feat_pca is not None:
                X_calib_feat = self._feat_pca.transform(X_calib_feat).astype(np.float32)

            self._cld_model = self._fit_stage2(
                X_src_feat, y_src, X_calib_feat, y_calib,
                stage1_model, mu, sigma, n_classes, n_neurons, source_cache,
            )
        else:
            self._cld_model = stage1_model

        self._fit_time = time.time() - t0
        return self

    def _fit_stage2(self, X_src_feat, y_src, X_calib_feat, y_calib,
                    stage1_model, mu, sigma, n_classes, n_neurons, source_cache=None):
        """Stage-2 solve (overridable hook). Base: source-anchored warm-start on
        source∪weighted-calibration. Subclasses can swap in a different stage-2 objective."""
        return fit_stage2_anchored(
            X_src_feat, y_src, X_calib_feat, y_calib, stage1_model, mu, sigma,
            n_classes=n_classes, n_neurons=n_neurons,
            rank=self.rank, beta=self.beta, rho=self.rho, gamma_ratio=self.gamma_ratio,
            admm_iters=self.admm_iters_stage2, pcg_iters=self.pcg_iters,
            seed=self.seed, target_mass=self.target_mass,
        )

    def _get_features(self, X: np.ndarray) -> np.ndarray:
        backbone = self._backbone_model.to(self.device)
        X_feat = extract_foundation_features(backbone, X, self.device, self.batch_size)
        if self._feat_pca is not None:
            X_feat = self._feat_pca.transform(X_feat).astype(np.float32)
        return X_feat

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None or self._backbone_model is None:
            raise RuntimeError("Adapter not fitted")
        X_feat = self._get_features(X)
        return _predict_from_cld(self._cld_model, self._feat_mu, self._feat_sigma, X_feat).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None or self._backbone_model is None:
            raise RuntimeError("Adapter not fitted")
        X_feat = self._get_features(X)
        logits = _predict_from_cld(self._cld_model, self._feat_mu, self._feat_sigma, X_feat)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


class FoundationSFTAnchoredEACLDAdapter(FoundationSFTAnchoredCLDAdapter):
    """EA whitening + SFT backbone + source-anchored 2-stage convex CLD head."""

    # Tune separately from the non-EA variant: EA whitens the feature space, so
    # its optimal (beta, target_mass) can differ. Distinct HP cache file.
    _hp_variant_tag: str = "ea_"

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 epsilon: float = 1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(
        self,
        source_data,
        target_unlabeled=None,
        target_labeled=None,
        source_cache: dict | None = None,
        source_per_subject: list | None = None,
    ) -> "FoundationSFTAnchoredEACLDAdapter":
        if source_data is None:
            raise ValueError("FoundationSFTAnchoredEACLDAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationSFTAnchoredEACLDAdapter requires target_unlabeled for EA")

        self._seed()
        X_src, y_src = source_data

        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Source alignment is identical across K/repeats — cache it so the full
        # source isn't re-whitened on every one of the 35 fits in a sweep.
        _ea_src_key = "sft_anchored_ea_src_aligned"
        if source_cache is not None and _ea_src_key in source_cache:
            X_src_aligned = source_cache[_ea_src_key]
        else:
            if source_per_subject is not None:
                aligned_chunks = []
                for X_subj, _ in source_per_subject:
                    R = compute_mean_covariance(X_subj, self.epsilon)
                    aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
                X_src_aligned = np.concatenate(aligned_chunks, axis=0)
            else:
                R_src = compute_mean_covariance(X_src, self.epsilon)
                X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))
            if source_cache is not None:
                source_cache[_ea_src_key] = X_src_aligned

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSFTAnchoredEACLDAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSFTAnchoredEACLDAdapter not fitted")
        return super().predict_proba(euclidean_align(X, self._target_R_inv_sqrt))
