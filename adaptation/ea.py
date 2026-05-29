"""Euclidean Alignment (EA) adapter.

He & Wu 2020, IEEE TBME: "Transfer Learning for Brain-Computer Interfaces:
A Euclidean Space Data Alignment Approach."

EA whitens each subject's trials by their mean covariance matrix so that
all subjects share a common covariance structure. Requires only unlabeled
target trials (K=0 unsupervised).
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter, train_epoch, evaluate_model


def compute_mean_covariance(X: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Compute mean covariance matrix across trials.

    Args:
        X: (N, C, T) trials
    Returns:
        R_mean: (C, C) mean covariance
    """
    covs = []
    for trial in X:
        # trial: (C, T)
        cov = trial @ trial.T / trial.shape[-1]
        covs.append(cov)
    R_mean = np.mean(covs, axis=0)
    # Add epsilon to diagonal for numerical stability
    R_mean += epsilon * np.eye(R_mean.shape[0])
    return R_mean


def matrix_sqrt_inv(M: np.ndarray) -> np.ndarray:
    """Compute M^{-1/2} via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-10)
    sqrt_inv = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    return sqrt_inv


def euclidean_align(X: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:
    """Apply EA transform: X_aligned[i] = R^{-1/2} @ X[i].

    Args:
        X: (N, C, T)
        R_inv_sqrt: (C, C) whitening matrix
    Returns:
        X_aligned: (N, C, T) float32
    """
    return (R_inv_sqrt @ X).astype(np.float32)


class EAAdapter(BaseAdapter):
    """Euclidean Alignment + LOSO classifier.

    Unsupervised (K=0): aligns target subject using unlabeled trials.
    Also aligns each source subject independently.
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 lr: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 200,
                 batch_size: int = 64, patience: int = 20, val_fraction: float = 0.1,
                 epsilon: float = 1e-6):
        super().__init__(backbone, device, seed)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.epsilon = epsilon
        self._model: nn.Module | None = None
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_per_subject: list | None = None) -> "EAAdapter":
        if source_data is None:
            raise ValueError("EAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("EAAdapter requires target_unlabeled for unsupervised alignment")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data

        # Per He & Wu 2020: align each source subject independently before pooling.
        # If per-subject data is provided, align each subject separately.
        # Otherwise fall back to pooled alignment (approximation).
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                R_inv_sqrt = matrix_sqrt_inv(R)
                aligned_chunks.append(euclidean_align(X_subj, R_inv_sqrt))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            R_src_inv_sqrt = matrix_sqrt_inv(R_src)
            X_src_aligned = euclidean_align(X_src, R_src_inv_sqrt)

        # Compute target alignment transform from unlabeled target trials
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        model = self._clone_backbone().to(self.device)

        # Train on aligned source data
        n_val = max(1, int(len(X_src_aligned) * self.val_fraction))
        idx = np.random.permutation(len(X_src_aligned))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr, y_tr = X_src_aligned[train_idx], y_src[train_idx]
        X_val, y_val = X_src_aligned[val_idx], y_src[val_idx]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
        self._model = model
        self._fit_time = time.time() - t0
        return self

    def _align_target(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            return X
        return euclidean_align(X, self._target_R_inv_sqrt)

    def _get_inference_model(self) -> nn.Module:
        return self._model if self._model is not None else self.backbone

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aligned = self._align_target(X)
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_aligned), 64):
                xb = torch.FloatTensor(X_aligned[start: start + 64]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aligned = self._align_target(X)
        return super().predict_proba(X_aligned)
