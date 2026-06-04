"""Source-anchored 2-stage convex CLD for specialist (non-foundation) backbones.

Specialist analogue of foundation_sft_anchored_cld.py. The convex 2-stage solve,
warm-startable ADMM and leak-free hyperparameter selection are backbone-agnostic
(they act on penultimate-feature matrices), so they are imported and reused here;
only the backbone handling differs:

  - Specialist: train the backbone from scratch on pooled source subjects
                (same as CLDAdapter), then read penultimate features via a forward
                hook on the last Linear/Conv2d layer.
  - Foundation: load a frozen source-fine-tuned backbone + foundation features.

Pipeline (per held-out subject):
  1. Source training  — fit backbone on pooled source (cached across K by seed).
  2. Stage 1          — cold ADMM on source features → CVX_ReLU_MLP.
  3. Stage 2 (K > 0)  — warm ADMM on source + weighted calibration, anchored to
                        the Stage 1 primal. K = 0 uses the Stage 1 model directly.

EAAnchoredCLDAdapter additionally whitens raw EEG trials with Euclidean Alignment
(unsupervised, from unlabeled target) before the backbone, mirroring EACLDAdapter.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model
from .cld import extract_penultimate_features, maybe_reduce_features
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_sft_anchored_cld import (
    _AnchoredHPSelectMixin,
    fit_stage1_source,
    fit_stage2_anchored,
    _predict_from_cld,
)
from models.foundations import FoundationBackbone


class AnchoredCLDAdapter(_AnchoredHPSelectMixin, BaseAdapter):
    """Source-trained specialist backbone + source-anchored 2-stage convex CLD head.

    Stage 1 (always): fit CVX_ReLU_MLP on source features via cold ADMM.
    Stage 2 (K > 0):  refit on source + weighted calibration, warm-starting the
                      Stage 1 primal variables (u, v).
    K = 0:            Stage 1 model used directly (zero-shot).
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        # Source-training params
        lr_src: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_src: int = 200,
        patience_src: int = 20,
        val_fraction_src: float = 0.1,
        batch_size: int = 64,
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
        if isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"AnchoredCLDAdapter (method 'anchored_cld') trains the backbone from "
                f"scratch on source and is for SPECIALIST backbones "
                f"(eegnet/shallowconv/conformer), but got a frozen "
                f"{type(backbone).__name__}. For foundation backbones use "
                f"'foundation_sft_anchored_cld' (FoundationSFTAnchoredCLDAdapter)."
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
        self.n_neurons = n_neurons  # None → auto: 10 binary, 32 multiclass
        self.admm_iters_stage2 = admm_iters_stage2
        self.target_mass = target_mass
        self.max_feat_dim = max_feat_dim
        self.hp_select = hp_select
        self.beta_grid = beta_grid
        self.target_mass_grid = target_mass_grid
        self.hp_val_k = hp_val_k

        self._backbone_model: nn.Module | None = None
        self._cld_model = None
        self._feat_mu: np.ndarray | None = None
        self._feat_sigma: np.ndarray | None = None
        self._feat_pca = None

    def _train_source(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """Train the backbone on pooled source with early stopping (cf. CLDAdapter)."""
        n_val = max(1, int(len(X) * self.val_fraction_src))
        idx = np.random.permutation(len(X))
        X_tr, y_tr = X[idx[n_val:]], y[idx[n_val:]]
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr_src, weight_decay=self.weight_decay
        )
        best_val_acc, best_state, patience_counter = -1.0, None, 0
        for _ in range(self.max_epochs_src):
            train_epoch(model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_src:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def fit(
        self,
        source_data,
        target_unlabeled=None,
        target_labeled=None,
        source_cache: dict | None = None,
    ) -> "AnchoredCLDAdapter":
        if source_data is None:
            raise ValueError("AnchoredCLDAdapter requires source_data")

        self._seed()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))
        n_neurons = self.n_neurons or (10 if n_classes == 2 else 32)

        # ---- Source-trained backbone — cached across K by seed. One-time shared
        # source-side cost, built BEFORE the fit_time clock. -------------------
        _bk = "anchored_cld_backbone"
        model = self._clone_backbone().to(self.device)
        if source_cache is not None and _bk in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[_bk]))
        else:
            model = self._train_source(model, X_src, y_src)
            if source_cache is not None:
                source_cache[_bk] = copy.deepcopy(model.state_dict())
        model.eval()
        self._backbone_model = model

        # ---- Source features (+ optional PCA), cached across K ---------------
        _fk = "anchored_cld_src_feats"
        if source_cache is not None and _fk in source_cache:
            X_src_feat, self._feat_pca = source_cache[_fk]
        else:
            X_src_feat = extract_penultimate_features(model, X_src, self.device, self.batch_size)
            if X_src_feat is None:
                raise RuntimeError("AnchoredCLDAdapter: no Linear/Conv2d layer for features")
            X_src_feat, self._feat_pca = maybe_reduce_features(X_src_feat, self.max_feat_dim, self.seed)
            if source_cache is not None:
                source_cache[_fk] = (X_src_feat, self._feat_pca)

        # ---- Stage-2 HP selection + Stage 1, cached across K -----------------
        # (beta, target_mass) are picked once per fold on a leak-free source
        # validation split; Stage 1 is then fit with the chosen beta. Both are
        # one-time source-side costs, so they sit BEFORE the fit_time clock.
        has_calib = target_labeled is not None and len(target_labeled[0]) >= n_classes
        _sk = "anchored_cld_stage1"
        if source_cache is not None and _sk in source_cache:
            self.beta, self.target_mass, stage1_model, mu, sigma = source_cache[_sk]
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
                source_cache[_sk] = (self.beta, self.target_mass, stage1_model, mu, sigma)
        self._feat_mu = mu
        self._feat_sigma = sigma

        # ---- fit_time covers per-K target adaptation only (Stage 2) ----------
        t0 = time.time()
        if has_calib:
            X_calib, y_calib = target_labeled
            # Target forward pass — counted into train_time (added to the Stage-2
            # solve below) so the on-target cost matches finetune/lora's boundary.
            _t_feat = time.perf_counter()
            X_calib_feat = extract_penultimate_features(model, X_calib, self.device, self.batch_size)
            if self._feat_pca is not None:
                X_calib_feat = self._feat_pca.transform(X_calib_feat).astype(np.float32)
            _feat_time = time.perf_counter() - _t_feat
            self._cld_model = self._fit_stage2(
                X_src_feat, y_src, X_calib_feat, y_calib,
                stage1_model, mu, sigma, n_classes, n_neurons, source_cache)
            # _fit_stage2 sets self._train_time to the Stage-2 solve time; add the
            # target forward pass so train_time = forward + solve.
            self._train_time = _feat_time + float(getattr(self, "_train_time", 0.0))
        else:
            # K=0: Stage 1 (source) model used directly — no target training.
            self._cld_model = stage1_model
            self._train_time = 0.0

        self._fit_time = time.time() - t0
        return self

    def _fit_stage2(self, X_src_feat, y_src, X_calib_feat, y_calib,
                    stage1_model, mu, sigma, n_classes, n_neurons, source_cache=None):
        """Stage-2 solve (overridable hook). Base: source-anchored warm-start on
        source∪weighted-calibration. Subclasses can swap in a different stage-2 objective
        (cf. foundation_sft_anchored_cld.FoundationSFTAnchoredCLDAdapter._fit_stage2)."""
        m = fit_stage2_anchored(
            X_src_feat, y_src, X_calib_feat, y_calib, stage1_model, mu, sigma,
            n_classes=n_classes, n_neurons=n_neurons,
            rank=self.rank, beta=self.beta, rho=self.rho, gamma_ratio=self.gamma_ratio,
            admm_iters=self.admm_iters_stage2, pcg_iters=self.pcg_iters,
            seed=self.seed, target_mass=self.target_mass,
        )
        self._train_time = float(getattr(m, "_solve_time", 0.0))
        return m

    def _get_features(self, X: np.ndarray) -> np.ndarray:
        X_feat = extract_penultimate_features(
            self._backbone_model, X, self.device, self.batch_size)
        if self._feat_pca is not None:
            X_feat = self._feat_pca.transform(X_feat).astype(np.float32)
        return X_feat

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None or self._backbone_model is None:
            raise RuntimeError("AnchoredCLDAdapter not fitted")
        X_feat = self._get_features(X)
        return _predict_from_cld(self._cld_model, self._feat_mu, self._feat_sigma, X_feat).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None or self._backbone_model is None:
            raise RuntimeError("AnchoredCLDAdapter not fitted")
        X_feat = self._get_features(X)
        logits = _predict_from_cld(self._cld_model, self._feat_mu, self._feat_sigma, X_feat)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)


class EAAnchoredCLDAdapter(AnchoredCLDAdapter):
    """EA whitening + source-trained backbone + source-anchored 2-stage CLD head.

    Mirrors EACLDAdapter: an unsupervised Euclidean Alignment transform is built
    from unlabeled target trials, source/target raw EEG are whitened, and the
    backbone + anchored CLD pipeline then runs on the aligned data.
    """

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
    ) -> "EAAnchoredCLDAdapter":
        if source_data is None:
            raise ValueError("EAAnchoredCLDAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("EAAnchoredCLDAdapter requires target_unlabeled for EA")

        self._seed()
        X_src, y_src = source_data

        # EA transform from unlabeled target trials.
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Align source per-subject (He & Wu 2020) when subject grouping is given.
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            X_src_aligned = euclidean_align(
                X_src, matrix_sqrt_inv(compute_mean_covariance(X_src, self.epsilon)))

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
            raise RuntimeError("EAAnchoredCLDAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("EAAnchoredCLDAdapter not fitted")
        return super().predict_proba(euclidean_align(X, self._target_R_inv_sqrt))
