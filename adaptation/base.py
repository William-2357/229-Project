"""Base adapter interface for all adaptation methods."""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Common interface for all adaptation methods.

    All adapters expose:
        fit(source_data, target_unlabeled, target_labeled)
        predict(X) -> np.ndarray of class indices
        predict_proba(X) -> np.ndarray (N, n_classes)
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42):
        self.backbone = backbone
        self.device = torch.device(device)
        self.seed = seed
        self._fit_time: float = 0.0

    def _seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _clone_backbone(self) -> nn.Module:
        return copy.deepcopy(self.backbone)

    @abstractmethod
    def fit(
        self,
        source_data: tuple[np.ndarray, np.ndarray] | None,
        target_unlabeled: np.ndarray | None,
        target_labeled: tuple[np.ndarray, np.ndarray] | None,
    ) -> "BaseAdapter":
        """Adapt the model.

        Args:
            source_data: (X_source, y_source) from N-1 subjects, or None
            target_unlabeled: X_target without labels (all target trials), or None
            target_labeled: (X_cal, y_cal) K-minute calibration set, or None
        Returns self for chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for (N, C, T) input."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities (N, n_classes)."""
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            logits = model(X_t)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def _get_inference_model(self) -> nn.Module:
        """Override in subclasses that store an adapted copy."""
        return self.backbone

    @property
    def fit_time(self) -> float:
        return self._fit_time

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(X).to(self.device)

    def _to_label_tensor(self, y: np.ndarray) -> torch.Tensor:
        return torch.LongTensor(y).to(self.device)


def train_epoch(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """One training epoch; returns mean cross-entropy loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    idx = np.random.permutation(len(X))
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(X), batch_size):
        batch_idx = idx[start: start + batch_size]
        xb = torch.FloatTensor(X[batch_idx]).to(device)
        yb = torch.LongTensor(y[batch_idx]).to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Return accuracy on (X, y)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[start: start + batch_size]).to(device)
            yb = y[start: start + batch_size]
            preds = model(xb).argmax(dim=-1).cpu().numpy()
            correct += (preds == yb).sum()
            total += len(yb)
    return correct / total if total > 0 else 0.0
