"""Euclidean Alignment adapter for pretrained foundation EEG backbones.

Mirrors EAAdapter but replaces backbone source-training with a linear probe
on the frozen pretrained foundation backbone.

K=0 unsupervised method: requires only unlabeled target trials for EA alignment.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from models.foundations import FoundationBackbone, FoundationWithHead


class _HeadWrapper(nn.Module):
    """Thin nn.Module that applies a linear head to pre-extracted feature vectors."""
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FoundationEAAdapter(BaseAdapter):
    """EA alignment + linear probe on a frozen pretrained foundation backbone.

    Workflow:
        1. Align source subjects (per-subject if source_per_subject provided)
        2. Align target using unlabeled target trials
        3. Train linear head on aligned source features (backbone frozen)
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        lr_probe: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_probe: int = 100,
        patience_probe: int = 15,
        val_fraction_probe: float = 0.1,
        batch_size: int = 32,
        epsilon: float = 1e-6,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationEAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_probe = lr_probe
        self.weight_decay = weight_decay
        self.max_epochs_probe = max_epochs_probe
        self.patience_probe = patience_probe
        self.val_fraction_probe = val_fraction_probe
        self.batch_size = batch_size
        self.epsilon = epsilon
        self._model: FoundationWithHead | None = None
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
        source_per_subject: list | None = None,
    ) -> "FoundationEAAdapter":
        if source_data is None:
            raise ValueError("FoundationEAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationEAAdapter requires target_unlabeled for EA alignment")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Step 1: Align source — per-subject if available (correct per He & Wu 2020)
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))

        # Step 2: Compute target alignment transform from unlabeled target trials
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Step 3: Linear probe on aligned source (backbone frozen)
        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)
        model.freeze_backbone()

        # Pre-extract features once (backbone frozen, so features are constant across epochs)
        model.eval()
        feats = []
        with torch.no_grad():
            for start in range(0, len(X_src_aligned), self.batch_size):
                xb = torch.FloatTensor(X_src_aligned[start: start + self.batch_size]).to(self.device)
                feats.append(model.backbone.get_features(xb).cpu().numpy())
        X_feat = np.concatenate(feats, axis=0)

        n_val = max(1, int(len(X_feat) * self.val_fraction_probe))
        idx = np.random.permutation(len(X_feat))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_feat[train_idx], y_src[train_idx]
        X_val, y_val = X_feat[val_idx], y_src[val_idx]

        head = _HeadWrapper(model.head)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=self.lr_probe, weight_decay=self.weight_decay
        )
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs_probe):
            train_epoch(head, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(head, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(head.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_probe:
                break

        if best_state is not None:
            head.load_state_dict(best_state)
        # head.head IS model.head (same object) — no copy needed
        self._model = model
        self._fit_time = time.time() - t0
        return self

    def _align_target(self, X: np.ndarray) -> np.ndarray:
        return euclidean_align(X, self._target_R_inv_sqrt)

    def _get_inference_model(self) -> nn.Module:
        return self._model if self._model is not None else self.backbone

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEAAdapter not fitted")
        X_aligned = self._align_target(X)
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_aligned), self.batch_size):
                xb = torch.FloatTensor(X_aligned[start: start + self.batch_size]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEAAdapter not fitted")
        return super().predict_proba(self._align_target(X))
