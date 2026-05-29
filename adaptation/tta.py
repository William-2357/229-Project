"""Test-Time Adaptation (TTA) adapter.

Auto-detects backbone architecture:
  - BatchNorm present → TENT (Wang et al. ICLR 2021): entropy minimization
    over BN affine params only.
  - No BatchNorm (LayerNorm only) → T3A (Iwasawa & Matsuo NeurIPS 2021):
    prototype adjustment using high-confidence unlabeled predictions.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter, train_epoch, evaluate_model
from models.specialists import has_batchnorm


# ---------------------------------------------------------------------------
# TENT helpers
# ---------------------------------------------------------------------------

def configure_tent(model: nn.Module) -> nn.Module:
    """Freeze all params except BN affine (gamma, beta)."""
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            # Use running stats from source training, update with target batches
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def tent_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Entropy minimization loss for TENT."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def run_tent(
    model: nn.Module,
    X_unlabeled: np.ndarray,
    device: torch.device,
    n_steps: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> nn.Module:
    """Run TENT: update BN affine params on unlabeled target data."""
    model = configure_tent(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    for _ in range(n_steps):
        perm = np.random.permutation(len(X_unlabeled))
        for start in range(0, len(X_unlabeled), batch_size):
            idx = perm[start: start + batch_size]
            xb = torch.FloatTensor(X_unlabeled[idx]).to(device)
            model.train()
            logits = model(xb)
            loss = tent_entropy_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# T3A helpers
# ---------------------------------------------------------------------------

def get_penultimate_features(model: nn.Module, X: np.ndarray, device: torch.device,
                             batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate layer features and logits by hooking the final Linear layer."""
    final_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            final_linear = m

    if final_linear is None:
        return None, None

    features_list = []
    hook = final_linear.register_forward_hook(
        lambda m, inp, out: features_list.append(inp[0].detach().cpu())
    )

    logits_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[start: start + batch_size]).to(device)
            logits_list.append(model(xb).cpu())

    hook.remove()
    features = torch.cat(features_list, dim=0).numpy()
    logits = torch.cat(logits_list, dim=0).numpy()
    return features, logits


def run_t3a(
    model: nn.Module,
    X_unlabeled: np.ndarray,
    device: torch.device,
    n_classes: int,
    confidence_threshold: float = 0.9,
    batch_size: int = 64,
) -> nn.Module:
    """T3A: adjust classifier prototypes using high-confidence unlabeled preds.

    Replaces the final linear classifier weights with mean penultimate-layer
    embeddings of high-confidence unlabeled predictions per class.
    """
    # Extract penultimate features and predictions
    features, logits = get_penultimate_features(model, X_unlabeled, device, batch_size)
    if features is None:
        return model

    probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
    conf = probs.max(axis=-1)
    pseudo = probs.argmax(axis=-1)

    # Collect high-confidence penultimate features per class
    embeddings_by_class: dict[int, list] = {c: [] for c in range(n_classes)}
    for i, (c, p) in enumerate(zip(pseudo, conf)):
        if p >= confidence_threshold:
            embeddings_by_class[int(c)].append(features[i])

    # Find final Linear layer and replace weights with class prototypes
    final_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            final_linear = m

    if final_linear is None:
        return model

    with torch.no_grad():
        new_weight = final_linear.weight.clone()
        for c in range(n_classes):
            if embeddings_by_class[c]:
                proto = np.mean(embeddings_by_class[c], axis=0)
                new_weight[c] = torch.FloatTensor(proto).to(device)
        final_linear.weight.copy_(new_weight)

    return model


# ---------------------------------------------------------------------------
# TTA Adapter
# ---------------------------------------------------------------------------

class TTAAdapter(BaseAdapter):
    """Test-Time Adaptation: TENT (BN models) or T3A (non-BN models).

    Unsupervised (K=0): uses only unlabeled target trials at test time.
    Requires a pre-trained source model (from LOSO training).
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 # Source training params
                 lr_src: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 200, batch_size: int = 64,
                 patience: int = 20, val_fraction: float = 0.1,
                 # TTA params
                 tent_steps: int = 10, tent_lr: float = 1e-3,
                 t3a_confidence: float = 0.9):
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.tent_steps = tent_steps
        self.tent_lr = tent_lr
        self.t3a_confidence = t3a_confidence
        self._model: nn.Module | None = None
        self._use_tent: bool | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None) -> "TTAAdapter":
        if source_data is None:
            raise ValueError("TTAAdapter requires source_data for pre-training")
        if target_unlabeled is None:
            raise ValueError("TTAAdapter requires target_unlabeled for TTA")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        model = self._clone_backbone().to(self.device)

        # Step 1: Train on source data (LOSO)
        n_val = max(1, int(len(X_src) * self.val_fraction))
        idx = np.random.permutation(len(X_src))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = X_src[train_idx], y_src[train_idx]
        X_val, y_val = X_src[val_idx], y_src[val_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_src, weight_decay=self.weight_decay)
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs):
            train_epoch(model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Step 2: Apply TTA on unlabeled target trials
        # Use TENT only for pure-BN architectures. Transformer models (e.g.
        # EEG-Conformer) have one incidental BN in the patch embedding but are
        # LayerNorm-dominant — TENT on a single BN layer destabilizes them.
        has_ln = any(isinstance(m, nn.LayerNorm) for m in model.modules())
        self._use_tent = has_batchnorm(model) and not has_ln
        n_classes = len(np.unique(y_src))

        if self._use_tent:
            model = run_tent(model, target_unlabeled, self.device,
                             n_steps=self.tent_steps, lr=self.tent_lr)
        else:
            model = run_t3a(model, target_unlabeled, self.device,
                            n_classes=n_classes, confidence_threshold=self.t3a_confidence)

        self._model = model
        self._fit_time = time.time() - t0
        return self

    def _get_inference_model(self) -> nn.Module:
        return self._model if self._model is not None else self.backbone

    def predict(self, X: np.ndarray) -> np.ndarray:
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), 64):
                xb = torch.FloatTensor(X[start: start + 64]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)
