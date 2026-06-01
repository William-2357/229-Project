"""Convex calibration adapter — the ONE file the autoresearch loop edits.

Research surface for "convex NN for low-resource EEG calibration" (research/program.md).
Backbone: a source-fine-tuned MIRepNet foundation encoder. Baselines (sft_lora/sft_finetune)
use the SAME disk-cached source-FT backbone, so comparisons are apples-to-apples.

iter-8 — LoRA + convex head (hybrid)
------------------------------------
Honest full-9 result: frozen-backbone convex (iter-3) trails LoRA because LoRA ADAPTS THE
REPRESENTATION at high K. So: do exactly sft_lora's LoRA adaptation on calibration (gentle,
low-rank — gentler than the full-finetune that diluted in iter-5), then replace LoRA's LINEAR
head with the convex ReLU head fit on source ∪ upweighted-calibration of the LoRA-adapted
features. This tests whether the convex head beats a linear head on the SAME representation —
i.e. whether LoRA+convex > LoRA. `use_lora=False` recovers iter-3 (frozen + convex on src∪cal).
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

import jax.numpy as jnp

from peft import LoraConfig, get_peft_model

from .base import BaseAdapter, train_epoch, evaluate_model
from .cld import fit_cld_head, maybe_reduce_features
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_cld import extract_foundation_features
from .foundation_lora import _get_lora_target_modules
from .foundation_source_finetune import build_source_finetuned_foundation_model
from models.foundations import FoundationBackbone, FoundationWithHead

# ---------------------------------------------------------------------------
# HPARAMS — the loop's primary tuning surface. Keep flat and documented.
# ---------------------------------------------------------------------------
HPARAMS = dict(
    # --- source fine-tuning of the FM backbone (shared with baselines; disk-cached) ---
    lr_src=1e-3, weight_decay=1e-4, max_epochs_src=200, patience_src=25,
    val_fraction_src=0.1, ft_batch_size=32,

    # --- front-end ---
    use_ea=False, ea_epsilon=1e-6,

    # --- convex ReLU head (jaxcld CVX_ReLU_MLP + ADMM) ---
    n_neurons=32, rank=20, beta=1e-4, rho=0.01, gamma_ratio=1.0,
    admm_iters=50, pcg_iters=10, max_feat_dim=None,

    # --- combined source ∪ upweighted calibration fit (iter-3) ---
    source_cap=800, cal_balance=4.0,

    # --- iter-8: LoRA representation adaptation before the convex head ---
    use_lora=True,
    lora_rank=8,            # matches the sft_lora baseline
    lora_lr=1e-3, lora_epochs=100, lora_patience=15, lora_val_frac=0.1,
    lora_min_cal=2,         # apply LoRA whenever calibration exists (like sft_lora)
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
    """Source-FT MIRepNet (+ optional LoRA on calibration) + convex head on source∪cal."""

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42, **overrides):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError("ConvexCalibAdapter requires a FoundationBackbone (e.g. mirepnet)")
        super().__init__(backbone, device, seed)
        self.hp = {**HPARAMS, **overrides}
        self._backbone_model = None
        self._cld_model = None
        self._feat_mu = self._feat_sigma = None
        self._pca = None
        self._target_R_inv_sqrt = None

    def _align(self, X):
        if not self.hp["use_ea"] or self._target_R_inv_sqrt is None:
            return X
        return euclidean_align(X, self._target_R_inv_sqrt)

    def _source_ft(self, X_src, y_src, n_classes, source_cache):
        h = self.hp
        key = ("convex_calib_sft", self.seed)
        if source_cache is not None and key in source_cache:
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            model.load_state_dict(copy.deepcopy(source_cache[key]))
            return model
        model = build_source_finetuned_foundation_model(
            self.backbone, n_classes, X_src, y_src, device=self.device,
            lr_src=h["lr_src"], weight_decay=h["weight_decay"], max_epochs_src=h["max_epochs_src"],
            patience_src=h["patience_src"], val_fraction_src=h["val_fraction_src"],
            batch_size=h["ft_batch_size"], seed=self.seed)
        if source_cache is not None:
            source_cache[key] = copy.deepcopy(model.state_dict())
        return model

    def _lora_adapt(self, model: FoundationWithHead, X_cal, y_cal) -> FoundationWithHead:
        """sft_lora's adaptation: LoRA on backbone, fine-tune LoRA params on cal, merge.
        Returns a FoundationWithHead whose backbone has LoRA folded in (head = source head)."""
        h = self.hp
        if len(X_cal) < 2:
            return model
        targets = _get_lora_target_modules(model, rank=h["lora_rank"])
        if not targets:
            return model
        cfg = LoraConfig(r=h["lora_rank"], lora_alpha=h["lora_rank"] * 2,
                         target_modules=targets, lora_dropout=0.1, bias="none")
        lm = get_peft_model(model, cfg)
        n_val = max(1, int(len(X_cal) * h["lora_val_frac"]))
        idx = np.random.permutation(len(X_cal))
        val_idx, tr_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx
        Xtr, ytr, Xv, yv = X_cal[tr_idx], y_cal[tr_idx], X_cal[val_idx], y_cal[val_idx]
        trainable = [p for p in lm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=h["lora_lr"], weight_decay=h["weight_decay"])
        best, best_state, pat = -1.0, None, 0
        for _ in range(h["lora_epochs"]):
            train_epoch(lm, Xtr, ytr, opt, self.device, h["ft_batch_size"])
            acc = evaluate_model(lm, Xv, yv, self.device)
            if acc > best:
                best, best_state, pat = acc, copy.deepcopy(lm.state_dict()), 0
            else:
                pat += 1
                if pat >= h["lora_patience"]:
                    break
        if best_state is not None:
            lm.load_state_dict(best_state)
        return lm.merge_and_unload()   # fold LoRA into backbone weights -> FoundationWithHead

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

        # 2) source-FT (cached). iter-8: then LoRA-adapt on calibration (gentle), freeze.
        model = self._source_ft(X_src, y_src, n_classes, source_cache)
        n_cal = len(target_labeled[0]) if target_labeled is not None else 0
        adapted = h["use_lora"] and n_cal >= h["lora_min_cal"]
        if adapted:
            model = self._lora_adapt(model, target_labeled[0], target_labeled[1])
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone_model = model.backbone.to(self.device)

        # 3) features from the (LoRA-adapted) frozen backbone
        bs = h["ft_batch_size"]
        X_src_feat = extract_foundation_features(self._backbone_model, X_src, self.device, bs)
        X_src_feat, y_src_sub = _stratified_subsample(X_src_feat, y_src, h["source_cap"], rng)

        norm_stats = None
        if target_unlabeled is not None and len(target_unlabeled) >= 2:
            # cache unlabeled feats only when backbone is the shared frozen one (not LoRA-adapted)
            uk = ("convex_calib_tgt_feats", self.seed)
            if not adapted and source_cache is not None and uk in source_cache:
                X_unlab = source_cache[uk]
            else:
                X_unlab = extract_foundation_features(self._backbone_model, target_unlabeled, self.device, bs)
                if not adapted and source_cache is not None:
                    source_cache[uk] = X_unlab
            norm_stats = (X_unlab.mean(0, keepdims=True), X_unlab.std(0, keepdims=True) + 1e-8)

        # 4) convex head on source ∪ upweighted calibration of the (adapted) features
        if target_labeled is not None and len(target_labeled[0]) >= 2:
            X_cal_feat = extract_foundation_features(self._backbone_model, target_labeled[0], self.device, bs)
            y_cal = target_labeled[1]
            cal_mass = h["cal_balance"] * len(X_src_feat)
            reps = max(1, int(round(cal_mass / max(1, len(X_cal_feat)))))
            X_fit = np.concatenate([X_src_feat, np.tile(X_cal_feat, (reps, 1))], axis=0)
            y_fit = np.concatenate([y_src_sub, np.tile(y_cal, reps)], axis=0)
        else:
            X_fit, y_fit = X_src_feat, y_src_sub

        X_fit, self._pca = maybe_reduce_features(X_fit, h["max_feat_dim"], self.seed)
        self._cld_model, self._feat_mu, self._feat_sigma = fit_cld_head(
            X_fit, y_fit, n_classes, h["n_neurons"], h["rank"], h["beta"], h["rho"],
            h["gamma_ratio"], h["admm_iters"], h["pcg_iters"], self.seed,
            norm_stats=norm_stats if h["max_feat_dim"] is None else None)
        self._fit_time = time.time() - t0
        return self

    def _logits(self, feat):
        if self._pca is not None:
            feat, _ = maybe_reduce_features(feat, self.hp["max_feat_dim"], self.seed, pca=self._pca)
        Zn = ((feat - self._feat_mu) / self._feat_sigma).astype(np.float32)
        return np.array(self._cld_model.stacked_predict(
            jnp.array(Zn), self._cld_model.theta1, self._cld_model.theta2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat = extract_foundation_features(self._backbone_model, self._align(X), self.device, self.hp["ft_batch_size"])
        return self._logits(feat).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feat = extract_foundation_features(self._backbone_model, self._align(X), self.device, self.hp["ft_batch_size"])
        logits = self._logits(feat)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
