"""CLD (Convex Language Detection) adapter for EEG classification.

Freezes a source-pretrained backbone and fits a convex two-layer ReLU
classification head via ADMM on extracted penultimate features.

Paper: Feng, Tan & Pilanci (2026). Convex Low-resource Accent-Robust
Language Detection in Speech Recognition. ICML 2026.

Binary:     rank=20, beta=1e-3, rho=0.01, n_neurons=10
Multiclass: rank=20, beta=1e-3, rho=0.01, n_neurons=32
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

import jax
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_compilation_cache_dir", "/root/.cache/jax_xla")
import jax.numpy as jnp
from jaxcld.models.cvx_relu_mlp import CVX_ReLU_MLP
from jaxcld.optimizers.admm import admm as _run_admm

from .base import BaseAdapter, train_epoch, evaluate_model


def extract_penultimate_features(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray | None:
    """Hook the last Linear or Conv2d layer and return its input as (N, d) features.

    Returns None if the model has no hookable layer.
    """
    last_layer = None
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            last_layer = m

    if last_layer is None:
        return None

    captured = []
    hook = last_layer.register_forward_hook(
        lambda m, inp, out: captured.append(inp[0].detach().cpu())
    )

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[start: start + batch_size]).to(device)
            model(xb)

    hook.remove()

    feats = torch.cat(captured, dim=0)
    if feats.dim() > 2:
        feats = feats.flatten(start_dim=1)
    return feats.numpy().astype(np.float32)


def fit_cld_head(
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
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple["CVX_ReLU_MLP", np.ndarray, np.ndarray]:
    """Normalize features, build CVX_ReLU_MLP, run ADMM.

    Returns (fitted_model, feature_mean, feature_std).
    If norm_stats=(mu, sigma) is provided, those statistics are used instead
    of computing them from X_feat (useful when X_feat is a small labeled set).
    """
    if norm_stats is not None:
        mu, sigma = norm_stats
    else:
        mu = X_feat.mean(axis=0, keepdims=True)
        sigma = X_feat.std(axis=0, keepdims=True) + 1e-8
    X_norm = ((X_feat - mu) / sigma).astype(np.float32)

    X_jax = jnp.array(X_norm)
    y_jax = jnp.array(y.astype(np.int32))
    key = jax.random.PRNGKey(seed)

    cld = CVX_ReLU_MLP(
        X=X_jax, y=y_jax, n_classes=n_classes, P_S=n_neurons,
        beta=beta, rho=rho, seed=key,
    )
    cld.init_model()

    _run_admm(cld, {
        'rank': rank,
        'beta': beta,
        'gamma_ratio': gamma_ratio,
        'admm_iters': admm_iters,
        'pcg_iters': pcg_iters,
        'check_opt': False,
    })

    return cld, mu, sigma


class CLDAdapter(BaseAdapter):
    """CLD: convex ADMM head on frozen backbone penultimate features.

    For k=0: fits on pooled source features (zero-shot baseline).
    For k>0: fits on labeled target calibration features — the key
             sample-efficient adaptation without backbone fine-tuning.
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 lr_src: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs_src: int = 200, batch_size: int = 64,
                 patience_src: int = 20, val_fraction_src: float = 0.1,
                 rank: int = 20, beta: float = 1e-3, rho: float = 0.01,
                 gamma_ratio: float = 1.0, admm_iters: int = 50,
                 pcg_iters: int = 10, n_neurons: int | None = None):
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.batch_size = batch_size
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.rank = rank
        self.beta = beta
        self.rho = rho
        self.gamma_ratio = gamma_ratio
        self.admm_iters = admm_iters
        self.pcg_iters = pcg_iters
        self.n_neurons = n_neurons  # None → auto: 10 binary, 32 multiclass
        self._backbone_model: nn.Module | None = None
        self._cld_model: CVX_ReLU_MLP | None = None
        self._feat_mu: np.ndarray | None = None
        self._feat_sigma: np.ndarray | None = None

    def _train_source(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
        n_val = max(1, int(len(X) * self.val_fraction_src))
        idx = np.random.permutation(len(X))
        X_tr, y_tr = X[idx[n_val:]], y[idx[n_val:]]
        X_val, y_val = X[idx[:n_val]], y[idx[:n_val]]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_src, weight_decay=self.weight_decay)
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

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None) -> "CLDAdapter":
        if source_data is None:
            raise ValueError("CLDAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))
        n_neurons = self.n_neurons if self.n_neurons is not None else (10 if n_classes == 2 else 32)

        # Step 1: Source pre-training (cached across k values by seed)
        model = self._clone_backbone().to(self.device)
        cache_key = self.seed
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
        else:
            model = self._train_source(model, X_src, y_src)
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())
        self._backbone_model = model

        # Step 2: Choose features for CLD head fitting.
        # k=0: fit on source features (zero-shot baseline).
        # k>0: fit on target calibration features only — same information
        #       budget as LoRA/finetune for a fair comparison.
        if target_labeled is not None and len(target_labeled[0]) >= 2:
            X_fit, y_fit = target_labeled
        else:
            X_fit, y_fit = X_src, y_src

        X_feat = extract_penultimate_features(model, X_fit, self.device)
        if X_feat is None:
            raise RuntimeError("CLDAdapter: no Linear or Conv2d layer for feature extraction")

        # Step 3: Fit CLD head via ADMM
        self._cld_model, self._feat_mu, self._feat_sigma = fit_cld_head(
            X_feat, y_fit, n_classes, n_neurons,
            self.rank, self.beta, self.rho, self.gamma_ratio,
            self.admm_iters, self.pcg_iters, self.seed,
        )

        self._fit_time = time.time() - t0
        return self

    def _get_inference_model(self) -> nn.Module:
        return self._backbone_model if self._backbone_model is not None else self.backbone

    def _predict_from_features(self, X_feat: np.ndarray) -> np.ndarray:
        X_norm = ((X_feat - self._feat_mu) / self._feat_sigma).astype(np.float32)
        logits = self._cld_model.stacked_predict(
            jnp.array(X_norm), self._cld_model.theta1, self._cld_model.theta2
        )
        return np.array(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None or self._backbone_model is None:
            raise RuntimeError("CLDAdapter not fitted")
        X_feat = extract_penultimate_features(self._backbone_model, X, self.device)
        return self._predict_from_features(X_feat).argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._cld_model is None or self._backbone_model is None:
            raise RuntimeError("CLDAdapter not fitted")
        X_feat = extract_penultimate_features(self._backbone_model, X, self.device)
        logits = self._predict_from_features(X_feat)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)
