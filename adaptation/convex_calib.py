"""Convex calibration adapter — the ONE file the autoresearch loop edits.

Backbone: source-fine-tuned MIRepNet (shared, disk-cached). On top, a convex two-layer
ReLU head (jaxcld ADMM) fit on source∪upweighted-calibration features.

Best so far: iter-8 LoRA+convex (use_lora=True) — full-9 0.595 > lora 0.587.

iter-11+ — cross-subject-generality objectives (deep-research shortlist)
-----------------------------------------------------------------------
`generality_mode` selects an objective that explicitly targets CROSS-SUBJECT generality
("learn a general rule, then convex-adapt to a new subject"; the learner never sees subject
IDs at test). Operates on the FROZEN backbone (use_lora=False) to isolate the objective.
Per-subject source is provided by run_local via source_cache['source_per_subject'].
  - "meta_r2d2": meta-train a low-rank feature adapter via leave-one-source-subject-out
    episodes with a CLOSED-FORM differentiable RIDGE inner solve (R2D2/MetaOptNet family).
    Each episode = a source subject's K support trials → ridge → query loss → backprop adapter.
    Learns features where few-shot convex adaptation generalizes across subjects.
  - "group_dro": fit the convex head under a minimax-over-subjects (worst source-subject)
    objective via online exp-grad subject reweighting (convex-concave; convergence-guaranteed).
  - "irm": meta-train the adapter with the IRMv1 gradient-penalty so one linear classifier is
    ~simultaneously optimal across all source subjects (subject-invariant predictor).
  - "none": iter-8 behavior (frozen or LoRA + convex head on source∪cal).
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
import jax.numpy as jnp

from peft import LoraConfig, get_peft_model

from .base import BaseAdapter, train_epoch, evaluate_model
from .cld import fit_cld_head, maybe_reduce_features
from .convex_transfer import sample_gates, build_fixed_gate_model, anchored_admm
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_cld import extract_foundation_features
from .foundation_lora import _get_lora_target_modules
from .foundation_source_finetune import build_source_finetuned_foundation_model
from models.foundations import FoundationBackbone, FoundationWithHead

HPARAMS = dict(
    # --- source fine-tuning of the FM backbone (shared with baselines; disk-cached) ---
    lr_src=1e-3, weight_decay=1e-4, max_epochs_src=200, patience_src=25,
    val_fraction_src=0.1, ft_batch_size=32,
    use_ea=False, ea_epsilon=1e-6,
    # --- convex ReLU head ---
    n_neurons=32, rank=20, beta=1e-4, rho=0.01, gamma_ratio=1.0,
    admm_iters=50, pcg_iters=10, max_feat_dim=None,
    source_cap=800, cal_balance=4.0,
    # --- LoRA representation adaptation (iter-8 winner) ---
    use_lora=True, lora_rank=8, lora_lr=1e-3, lora_epochs=100, lora_patience=15,
    lora_val_frac=0.1, lora_min_cal=2,
    # --- two-stage convex transfer (relaxed-harness arc): fixed dict + anchor to source head ---
    transfer_mode="none",     # none | anchor   (anchor => convex-pretrain source head, then
                              #                   anchored target solve; see convex_transfer.py)
    anchor_a=0.01,            # quadratic anchor strength toward the source convex head v_bar
    transfer_stage2="cal",    # cal | source_cal : fit set for the anchored target solve
    # --- cross-subject-generality objective ---
    generality_mode="none",   # none | meta_r2d2 | group_dro | irm
    gen_adapter_rank=16,      # low-rank adapter A(f)=f+(f@U)@V for meta/irm
    # meta_r2d2
    meta_steps=400, meta_lr=1e-3, meta_wd=1e-4, meta_support_per_class=4,
    meta_ridge_lambda=1.0,
    # group_dro
    dro_rounds=4, dro_eta=2.0,
    # irm
    irm_steps=400, irm_lr=1e-3, irm_lambda=1.0, irm_wd=1e-4,
)


import os as _os, json as _json
_ov = _os.environ.get("CONVEX_HP")   # run a mode without editing the file: CONVEX_HP='{...}'
if _ov:
    HPARAMS.update(_json.loads(_ov))


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


def _ridge_W(Z, Y, lam):
    """Closed-form differentiable ridge: argmin_W ||ZW-Y||^2 + lam||W||^2. Woodbury when n<=d."""
    n, d = Z.shape
    if n <= d:
        G = Z @ Z.T + lam * torch.eye(n, device=Z.device)
        return Z.T @ torch.linalg.solve(G, Y)
    A = Z.T @ Z + lam * torch.eye(d, device=Z.device)
    return torch.linalg.solve(A, Z.T @ Y)


class ConvexCalibAdapter(BaseAdapter):
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
        self._U = self._V = None   # generality feature adapter (None => identity)

    # -- helpers ------------------------------------------------------------
    def _align(self, X):
        if not self.hp["use_ea"] or self._target_R_inv_sqrt is None:
            return X
        return euclidean_align(X, self._target_R_inv_sqrt)

    def _adapt_np(self, feat):
        if self._U is None:
            return feat
        U = self._U.cpu().numpy(); V = self._V.cpu().numpy()
        return (feat + (feat @ U) @ V).astype(np.float32)

    def _source_ft(self, X_src, y_src, n_classes, source_cache):
        h = self.hp
        key = ("convex_calib_sft", self.seed)
        if source_cache is not None and key in source_cache:
            m = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            m.load_state_dict(copy.deepcopy(source_cache[key])); return m
        m = build_source_finetuned_foundation_model(
            self.backbone, n_classes, X_src, y_src, device=self.device,
            lr_src=h["lr_src"], weight_decay=h["weight_decay"], max_epochs_src=h["max_epochs_src"],
            patience_src=h["patience_src"], val_fraction_src=h["val_fraction_src"],
            batch_size=h["ft_batch_size"], seed=self.seed)
        if source_cache is not None:
            source_cache[key] = copy.deepcopy(m.state_dict())
        return m

    def _lora_adapt(self, model, X_cal, y_cal):
        h = self.hp
        if len(X_cal) < 2:
            return model
        targets = _get_lora_target_modules(model, rank=h["lora_rank"])
        if not targets:
            return model
        lm = get_peft_model(model, LoraConfig(r=h["lora_rank"], lora_alpha=h["lora_rank"] * 2,
                            target_modules=targets, lora_dropout=0.1, bias="none"))
        n_val = max(1, int(len(X_cal) * h["lora_val_frac"])); idx = np.random.permutation(len(X_cal))
        vi, ti = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx
        opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad],
                                lr=h["lora_lr"], weight_decay=h["weight_decay"])
        best, bs, pat = -1.0, None, 0
        for _ in range(h["lora_epochs"]):
            train_epoch(lm, X_cal[ti], y_cal[ti], opt, self.device, h["ft_batch_size"])
            acc = evaluate_model(lm, X_cal[vi], y_cal[vi], self.device)
            if acc > best: best, bs, pat = acc, copy.deepcopy(lm.state_dict()), 0
            else:
                pat += 1
                if pat >= h["lora_patience"]: break
        if bs is not None: lm.load_state_dict(bs)
        return lm.merge_and_unload()

    def _subject_feats(self, source_cache, n_classes):
        """Per-source-subject (features, labels) from the frozen backbone (cached)."""
        sps = source_cache.get("source_per_subject")
        bs = self.hp["ft_batch_size"]
        feats, labels = [], []
        for Xs, ys in sps:
            feats.append(extract_foundation_features(self._backbone_model, self._maybe_ea(Xs), self.device, bs))
            labels.append(ys.astype(np.int64))
        return feats, labels

    def _maybe_ea(self, X):
        if self.hp["use_ea"] and self._target_R_inv_sqrt is not None:
            return euclidean_align(X, self._target_R_inv_sqrt)
        return X

    # -- generality objectives ---------------------------------------------
    def _meta_train_adapter(self, feats, labels, d, n_classes):
        """R2D2: meta-train A via per-subject few-shot episodes with closed-form ridge."""
        h = self.hp; dev = self.device
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        U = nn.Parameter((torch.randn(d, h["gen_adapter_rank"], generator=g) * 0.01).to(dev))
        V = nn.Parameter(torch.zeros(h["gen_adapter_rank"], d, device=dev))
        opt = torch.optim.AdamW([U, V], lr=h["meta_lr"], weight_decay=h["meta_wd"])
        fts = [torch.tensor(f, dtype=torch.float32, device=dev) for f in feats]
        lbs = [torch.tensor(l, dtype=torch.long, device=dev) for l in labels]
        spc = h["meta_support_per_class"]
        rng = np.random.default_rng(self.seed)
        for step in range(h["meta_steps"]):
            si = int(rng.integers(len(fts)))
            y = lbs[si]; f = fts[si]
            sup, qry = [], []
            for c in range(n_classes):
                ci = torch.where(y == c)[0].cpu().numpy()
                rng.shuffle(ci)
                sup.extend(ci[:spc]); qry.extend(ci[spc:spc + 24])
            sup = torch.tensor(sup, device=dev); qry = torch.tensor(qry, device=dev)
            Z = f + (f @ U) @ V
            Zs, Zq = Z[sup], Z[qry]
            Ys = F.one_hot(y[sup], n_classes).float()
            W = _ridge_W(Zs, Ys, h["meta_ridge_lambda"])
            loss = F.cross_entropy(Zq @ W, y[qry])
            opt.zero_grad(); loss.backward(); opt.step()
        self._U, self._V = U.detach(), V.detach()

    def _irm_train_adapter(self, feats, labels, d, n_classes):
        """IRMv1: train adapter + linear head with the dummy-scale gradient-norm penalty."""
        h = self.hp; dev = self.device
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        U = nn.Parameter((torch.randn(d, h["gen_adapter_rank"], generator=g) * 0.01).to(dev))
        V = nn.Parameter(torch.zeros(h["gen_adapter_rank"], d, device=dev))
        w = nn.Parameter((torch.randn(d, n_classes, generator=g) * 0.01).to(dev))
        opt = torch.optim.AdamW([U, V, w], lr=h["irm_lr"], weight_decay=h["irm_wd"])
        fts = [torch.tensor(f, dtype=torch.float32, device=dev) for f in feats]
        lbs = [torch.tensor(l, dtype=torch.long, device=dev) for l in labels]
        for step in range(h["irm_steps"]):
            ce_tot = 0.0; pen_tot = 0.0
            for f, y in zip(fts, lbs):
                Z = f + (f @ U) @ V
                scale = torch.ones(1, device=dev, requires_grad=True)
                logits = scale * (Z @ w)
                ce = F.cross_entropy(logits, y)
                gpen = torch.autograd.grad(ce, scale, create_graph=True)[0]
                ce_tot = ce_tot + ce; pen_tot = pen_tot + (gpen ** 2).sum()
            loss = ce_tot / len(fts) + h["irm_lambda"] * pen_tot / len(fts)
            opt.zero_grad(); loss.backward(); opt.step()
        self._U, self._V = U.detach(), V.detach()

    def _group_dro_fit(self, feats, labels, cal_feat, y_cal, norm, n_classes):
        """Fit convex head under worst-source-subject reweighting (online exp-grad)."""
        h = self.hp; rng = np.random.default_rng(self.seed)
        m = len(feats); q = np.ones(m) / m
        total = h["source_cap"]
        cld = mu = sig = None
        for r in range(h["dro_rounds"]):
            # build reweighted source pool: subject s contributes ~ q_s * total rows
            chunks_X, chunks_y = [], []
            for s in range(m):
                ns = max(n_classes, int(round(q[s] * total)))
                Xs, ys = _stratified_subsample(feats[s], labels[s], ns, rng)
                chunks_X.append(Xs); chunks_y.append(ys)
            Xsrc = np.concatenate(chunks_X); ysrc = np.concatenate(chunks_y)
            if cal_feat is not None:
                reps = max(1, int(round(h["cal_balance"] * len(Xsrc) / max(1, len(cal_feat)))))
                Xfit = np.concatenate([Xsrc, np.tile(cal_feat, (reps, 1))])
                yfit = np.concatenate([ysrc, np.tile(y_cal, reps)])
            else:
                Xfit, yfit = Xsrc, ysrc
            cld, mu, sig = fit_cld_head(Xfit, yfit, n_classes, h["n_neurons"], h["rank"], h["beta"],
                                        h["rho"], h["gamma_ratio"], h["admm_iters"], h["pcg_iters"],
                                        self.seed, norm_stats=norm)
            if r == h["dro_rounds"] - 1:
                break
            # per-subject loss -> exp-grad reweight (upweight worst subjects)
            losses = []
            for s in range(m):
                Xn = ((feats[s] - mu) / sig).astype(np.float32)
                logits = np.array(cld.stacked_predict(jnp.array(Xn), cld.theta1, cld.theta2))
                p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                ll = -np.log(p[np.arange(len(labels[s])), labels[s]] + 1e-9).mean()
                losses.append(ll)
            losses = np.array(losses)
            q = q * np.exp(h["dro_eta"] * (losses - losses.mean()))
            q = np.clip(q, 1e-4, None); q /= q.sum()
        return cld, mu, sig

    # -- two-stage convex transfer (fixed dictionary + anchor to source head) ----
    def _admm_params(self):
        h = self.hp
        return {'rank': h["rank"], 'beta': h["beta"], 'gamma_ratio': h["gamma_ratio"],
                'admm_iters': h["admm_iters"], 'pcg_iters': h["pcg_iters"], 'check_opt': False}

    def _transfer_head(self, X_src_feat, y_src, cal_feat, y_cal, norm, n_classes, source_cache):
        """Stage 1: convex-pretrain a source head v_bar on FIXED gates G (cached per subject).
        Stage 2: re-solve on the target calibration set with a quadratic anchor toward v_bar.
        Source knowledge enters ONLY via the shared gates + the anchor (not raw source pooling
        when transfer_stage2='cal') — the convex analog of pretrain->finetune."""
        h = self.hp
        if norm is None:
            norm = (X_src_feat.mean(0, keepdims=True), X_src_feat.std(0, keepdims=True) + 1e-8)
        mu, sigma = norm
        d = X_src_feat.shape[1]
        # Stage-1 anchor is cacheable across K/repeats ONLY on a frozen backbone; under LoRA
        # the source features differ per cell, so a cached v_bar would be stale -> recompute.
        ck = ("convex_transfer", self.seed)
        cacheable = source_cache is not None and not h["use_lora"]
        if cacheable and ck in source_cache:
            G, v_bar = source_cache[ck]
        else:
            key = jax.random.PRNGKey(self.seed)
            G, key = sample_gates(d, h["n_neurons"], key)
            Xs_n = ((X_src_feat - mu) / sigma).astype(np.float32)
            src = build_fixed_gate_model(Xs_n, y_src, n_classes, h["n_neurons"],
                                         h["beta"], h["rho"], key, G)
            anchored_admm(src, self._admm_params(), v_anchor=None, anchor_a=0.0)
            v_bar = src.v
            if cacheable:
                source_cache[ck] = (G, v_bar)
        if h["transfer_stage2"] == "source_cal":
            reps = max(1, int(round(h["cal_balance"] * len(X_src_feat) / max(1, len(cal_feat)))))
            X_fit = np.concatenate([X_src_feat, np.tile(cal_feat, (reps, 1))], axis=0)
            y_fit = np.concatenate([y_src, np.tile(y_cal, reps)], axis=0)
        else:
            X_fit, y_fit = cal_feat, y_cal
        Xn = ((X_fit - mu) / sigma).astype(np.float32)
        tgt = build_fixed_gate_model(Xn, y_fit, n_classes, h["n_neurons"],
                                     h["beta"], h["rho"], jax.random.PRNGKey(self.seed + 1), G)
        anchored_admm(tgt, self._admm_params(), v_anchor=v_bar, anchor_a=h["anchor_a"])
        self._cld_model, self._feat_mu, self._feat_sigma = tgt, mu, sigma

    # -- BaseAdapter interface ---------------------------------------------
    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None, source_per_subject: list | None = None):
        if source_data is None:
            raise ValueError("ConvexCalibAdapter requires source_data")
        self._seed(); t0 = time.time(); rng = np.random.default_rng(self.seed)
        h = self.hp; mode = h["generality_mode"]
        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        if h["use_ea"]:
            eps = h["ea_epsilon"]
            self._target_R_inv_sqrt = matrix_sqrt_inv(compute_mean_covariance(target_unlabeled, eps))
            X_src = euclidean_align(X_src, matrix_sqrt_inv(compute_mean_covariance(X_src, eps)))
            target_unlabeled = euclidean_align(target_unlabeled, self._target_R_inv_sqrt)
            if target_labeled is not None:
                target_labeled = (euclidean_align(target_labeled[0], self._target_R_inv_sqrt), target_labeled[1])

        model = self._source_ft(X_src, y_src, n_classes, source_cache)
        n_cal = len(target_labeled[0]) if target_labeled is not None else 0
        if h["use_lora"] and mode == "none" and n_cal >= h["lora_min_cal"]:
            model = self._lora_adapt(model, target_labeled[0], target_labeled[1])
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone_model = model.backbone.to(self.device)
        bs = h["ft_batch_size"]; d = int(self.backbone.feature_dim)

        # cross-subject-generality adapter trained on per-subject SOURCE features.
        # Depends only on source (per target subject), NOT on K/cal -> cache across K/repeats.
        if mode in ("meta_r2d2", "irm"):
            ck = ("gen_adapter", mode, self.seed)
            if source_cache is not None and ck in source_cache:
                self._U, self._V = source_cache[ck]
            else:
                feats, labels = self._subject_feats(source_cache, n_classes)
                if mode == "meta_r2d2":
                    self._meta_train_adapter(feats, labels, d, n_classes)
                else:
                    self._irm_train_adapter(feats, labels, d, n_classes)
                if source_cache is not None:
                    source_cache[ck] = (self._U, self._V)

        # head-fit features (adapter applied)
        X_src_feat = self._adapt_np(extract_foundation_features(self._backbone_model, X_src, self.device, bs))
        X_src_feat, y_src_sub = _stratified_subsample(X_src_feat, y_src, h["source_cap"], rng)
        f_unlab = None
        if target_unlabeled is not None and len(target_unlabeled) >= 2:
            f_unlab = self._adapt_np(extract_foundation_features(self._backbone_model, target_unlabeled, self.device, bs))
            norm = (f_unlab.mean(0, keepdims=True), f_unlab.std(0, keepdims=True) + 1e-8)
        else:
            norm = None
        cal_feat = y_cal = None
        if target_labeled is not None and len(target_labeled[0]) >= 2:
            cal_feat = self._adapt_np(extract_foundation_features(self._backbone_model, target_labeled[0], self.device, bs))
            y_cal = target_labeled[1]

        if h["transfer_mode"] == "anchor" and cal_feat is not None:
            self._transfer_head(X_src_feat, y_src_sub, cal_feat, y_cal, norm, n_classes, source_cache)
        elif mode == "group_dro":
            sf, sl = self._subject_feats(source_cache, n_classes)
            sf = [self._adapt_np(x) for x in sf]   # identity (no adapter in dro)
            self._cld_model, self._feat_mu, self._feat_sigma = self._group_dro_fit(
                sf, sl, cal_feat, y_cal, norm, n_classes)
        else:
            if cal_feat is not None:
                reps = max(1, int(round(h["cal_balance"] * len(X_src_feat) / max(1, len(cal_feat)))))
                X_fit = np.concatenate([X_src_feat, np.tile(cal_feat, (reps, 1))], axis=0)
                y_fit = np.concatenate([y_src_sub, np.tile(y_cal, reps)], axis=0)
            else:
                X_fit, y_fit = X_src_feat, y_src_sub
            X_fit, self._pca = maybe_reduce_features(X_fit, h["max_feat_dim"], self.seed)
            self._cld_model, self._feat_mu, self._feat_sigma = fit_cld_head(
                X_fit, y_fit, n_classes, h["n_neurons"], h["rank"], h["beta"], h["rho"],
                h["gamma_ratio"], h["admm_iters"], h["pcg_iters"], self.seed,
                norm_stats=norm if h["max_feat_dim"] is None else None)
        self._fit_time = time.time() - t0
        return self

    def _logits(self, feat):
        feat = self._adapt_np(feat)
        if self._pca is not None:
            feat, _ = maybe_reduce_features(feat, self.hp["max_feat_dim"], self.seed, pca=self._pca)
        Zn = ((feat - self._feat_mu) / self._feat_sigma).astype(np.float32)
        return np.array(self._cld_model.stacked_predict(jnp.array(Zn), self._cld_model.theta1, self._cld_model.theta2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat = extract_foundation_features(self._backbone_model, self._align(X), self.device, self.hp["ft_batch_size"])
        return self._logits(feat).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feat = extract_foundation_features(self._backbone_model, self._align(X), self.device, self.hp["ft_batch_size"])
        logits = self._logits(feat)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
