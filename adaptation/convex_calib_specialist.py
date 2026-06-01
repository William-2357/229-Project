"""Specialist-backbone port of the convex calibration methods (eegnet/shallowconv/conformer).

Mirrors the foundation convex_calib for from-scratch specialist backbones:
  - source-train the specialist backbone on pooled source (cached), then freeze
  - mode "convex":      convex ReLU head (jaxcld ADMM) on source ∪ upweighted-calibration
                        penultimate features  (the iter-3 source∪cal fix — no low-K dip)
  - mode "lora_convex": LoRA-adapt the source-trained backbone on the calibration trials,
                        merge, THEN the same source∪cal convex head  (the iter-8 hybrid)

Reuses cld.extract_penultimate_features / fit_cld_head and the standard source-train loop.
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

import jax.numpy as jnp

from .base import BaseAdapter, train_epoch, evaluate_model
from .cld import extract_penultimate_features, fit_cld_head


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


class SpecialistConvexAdapter(BaseAdapter):
    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 mode: str = "convex",
                 lr_src=1e-3, weight_decay=1e-4, max_epochs_src=200, patience_src=20,
                 val_fraction_src=0.1, batch_size=64,
                 n_neurons=32, rank=20, beta=1e-4, rho=0.01, gamma_ratio=1.0,
                 admm_iters=50, pcg_iters=10, source_cap=800, cal_balance=4.0,
                 lr_tgt=1e-4, max_epochs_tgt=60, patience_tgt=12):
        super().__init__(backbone, device, seed)
        self.mode = mode   # "convex" (frozen source-trained) | "ft_convex" (FT-adapt on cal)
        self.lr_src = lr_src; self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src; self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src; self.batch_size = batch_size
        self.n_neurons = n_neurons; self.rank = rank; self.beta = beta; self.rho = rho
        self.gamma_ratio = gamma_ratio; self.admm_iters = admm_iters; self.pcg_iters = pcg_iters
        self.source_cap = source_cap; self.cal_balance = cal_balance
        # representation adaptation on calibration: gentle FULL fine-tune (peft-LoRA can't wrap
        # braindecode MaxNorm conv layers under torch 2.11 — parametrization clash).
        self.lr_tgt = lr_tgt; self.max_epochs_tgt = max_epochs_tgt; self.patience_tgt = patience_tgt
        self._model = None; self._cld_model = None
        self._feat_mu = self._feat_sigma = None

    def _train_source(self, model, X, y):
        n_val = max(1, int(len(X) * self.val_fraction_src))
        idx = np.random.permutation(len(X))
        X_tr, y_tr, X_val, y_val = X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr_src, weight_decay=self.weight_decay)
        best, best_state, pat = -1.0, None, 0
        for _ in range(self.max_epochs_src):
            train_epoch(model, X_tr, y_tr, opt, self.device, self.batch_size)
            acc = evaluate_model(model, X_val, y_val, self.device)
            if acc > best: best, best_state, pat = acc, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= self.patience_src: break
        if best_state is not None: model.load_state_dict(best_state)
        return model

    def _ft_adapt(self, model, X_cal, y_cal):
        """Gentle FULL fine-tune of the source-trained backbone on calibration (low lr,
        early-stop on a cal val split) — the specialist stand-in for LoRA representation
        adaptation. Returns the adapted model."""
        if len(X_cal) < 2: return model
        for p in model.parameters():
            p.requires_grad_(True)
        n_val = max(1, int(len(X_cal) * 0.1)); idx = np.random.permutation(len(X_cal))
        vi, ti = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr_tgt, weight_decay=self.weight_decay)
        best, bs, pat = -1.0, None, 0
        for _ in range(self.max_epochs_tgt):
            train_epoch(model, X_cal[ti], y_cal[ti], opt, self.device, self.batch_size)
            acc = evaluate_model(model, X_cal[vi], y_cal[vi], self.device)
            if acc > best: best, bs, pat = acc, copy.deepcopy(model.state_dict()), 0
            else:
                pat += 1
                if pat >= self.patience_tgt: break
        if bs is not None: model.load_state_dict(bs)
        return model

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject=None):
        if source_data is None:
            raise ValueError("requires source_data")
        self._seed(); t0 = time.time(); rng = np.random.default_rng(self.seed)
        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        model = self._clone_backbone().to(self.device)
        ck = self.seed
        if source_cache is not None and ck in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[ck]))
        else:
            model = self._train_source(model, X_src, y_src)
            if source_cache is not None:
                source_cache[ck] = copy.deepcopy(model.state_dict())

        if self.mode in ("ft_convex", "ft_linear") and target_labeled is not None and len(target_labeled[0]) >= 2:
            model = self._ft_adapt(model, target_labeled[0], target_labeled[1])
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

        # linear-head baselines (same backbone/source-train/FT) — predict via model logits,
        # NO convex head. ft_linear == finetune; frozen_linear == source-trained zero-shot.
        if self.mode in ("ft_linear", "frozen_linear"):
            self._cld_model = None
            self._fit_time = time.time() - t0
            return self

        Xs_feat = extract_penultimate_features(model, X_src, self.device)
        Xs_feat, ys_sub = _stratified_subsample(Xs_feat, y_src, self.source_cap, rng)
        norm = None
        if target_unlabeled is not None and len(target_unlabeled) >= 2:
            fu = extract_penultimate_features(model, target_unlabeled, self.device)
            norm = (fu.mean(0, keepdims=True), fu.std(0, keepdims=True) + 1e-8)
        if target_labeled is not None and len(target_labeled[0]) >= 2:
            fc = extract_penultimate_features(model, target_labeled[0], self.device)
            yc = target_labeled[1]
            reps = max(1, int(round(self.cal_balance * len(Xs_feat) / max(1, len(fc)))))
            X_fit = np.concatenate([Xs_feat, np.tile(fc, (reps, 1))], 0)
            y_fit = np.concatenate([ys_sub, np.tile(yc, reps)], 0)
        else:
            X_fit, y_fit = Xs_feat, ys_sub
        self._cld_model, self._feat_mu, self._feat_sigma = fit_cld_head(
            X_fit, y_fit, n_classes, self.n_neurons, self.rank, self.beta, self.rho,
            self.gamma_ratio, self.admm_iters, self.pcg_iters, self.seed, norm_stats=norm)
        self._fit_time = time.time() - t0
        return self

    def _logits(self, X):
        if self._cld_model is None:   # linear-head baseline: use the backbone's own logits
            self._model.eval(); self._model.to(self.device)
            outs = []
            with torch.no_grad():
                for s in range(0, len(X), self.batch_size):
                    xb = torch.FloatTensor(X[s:s + self.batch_size]).to(self.device)
                    outs.append(self._model(xb).cpu().numpy())
            return np.concatenate(outs, axis=0)
        feat = extract_penultimate_features(self._model, X, self.device)
        Zn = ((feat - self._feat_mu) / self._feat_sigma).astype(np.float32)
        return np.array(self._cld_model.stacked_predict(jnp.array(Zn), self._cld_model.theta1, self._cld_model.theta2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lo = self._logits(X); e = np.exp(lo - lo.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
