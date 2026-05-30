"""Test-Time Adaptation (T3A) for pretrained foundation EEG backbones.

Foundation backbones use LayerNorm throughout (no BatchNorm), so TENT's
BN-affine update doesn't apply. T3A (Iwasawa & Matsuo NeurIPS 2021) is used
instead: prototype adjustment via high-confidence unlabeled target predictions.

Workflow:
    1. Freeze backbone; train linear head on source features (linear probe)
    2. Extract features from unlabeled target trials using frozen backbone
    3. Replace head weights with class-mean prototypes from high-confidence preds
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter
from .linear_probe import train_linear_probe
from models.foundations import FoundationBackbone, FoundationWithHead


def _run_foundation_t3a(
    model: FoundationWithHead,
    X_unlabeled: np.ndarray,
    device: torch.device,
    n_classes: int,
    confidence_threshold: float = 0.9,
    batch_size: int = 64,
) -> FoundationWithHead:
    """Adjust head prototypes using high-confidence unlabeled target predictions.

    Uses backbone.get_features() directly instead of a forward hook, since
    FoundationWithHead exposes the feature extractor explicitly.
    """
    model.eval()

    feats, logits_list = [], []
    with torch.no_grad():
        for start in range(0, len(X_unlabeled), batch_size):
            xb = torch.FloatTensor(X_unlabeled[start: start + batch_size]).to(device)
            f = model.backbone.get_features(xb)
            feats.append(f.cpu())
            logits_list.append(model.head(f).cpu())

    features = torch.cat(feats, dim=0).numpy()        # (N, feat_dim)
    logits = torch.cat(logits_list, dim=0).numpy()    # (N, n_classes)

    probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
    conf = probs.max(axis=-1)
    pseudo = probs.argmax(axis=-1)

    embeddings_by_class: dict[int, list] = {c: [] for c in range(n_classes)}
    for i, (c, p) in enumerate(zip(pseudo, conf)):
        if p >= confidence_threshold:
            embeddings_by_class[int(c)].append(features[i])

    with torch.no_grad():
        new_weight = model.head.weight.clone()
        for c in range(n_classes):
            if embeddings_by_class[c]:
                proto = np.mean(embeddings_by_class[c], axis=0)
                new_weight[c] = torch.FloatTensor(proto).to(device)
        model.head.weight.copy_(new_weight)

    return model


class FoundationTTAAdapter(BaseAdapter):
    """T3A test-time adaptation for foundation backbones.

    Analogous to TTAAdapter for specialist models (T3A branch only — no TENT,
    since foundation backbones have no BatchNorm layers to adapt).
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
        t3a_confidence: float = 0.9,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationTTAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_probe = lr_probe
        self.weight_decay = weight_decay
        self.max_epochs_probe = max_epochs_probe
        self.patience_probe = patience_probe
        self.val_fraction_probe = val_fraction_probe
        self.batch_size = batch_size
        self.t3a_confidence = t3a_confidence
        self._model: FoundationWithHead | None = None

    def _probe_kwargs(self) -> dict:
        return dict(
            device=self.device,
            batch_size=self.batch_size,
            lr=self.lr_probe,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs_probe,
            patience=self.patience_probe,
            val_fraction=self.val_fraction_probe,
        )

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
    ) -> "FoundationTTAAdapter":
        if source_data is None:
            raise ValueError("FoundationTTAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationTTAAdapter requires target_unlabeled for T3A")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)

        # Step 1: Linear probe on source features (cached — identical across calls)
        cache_key = ("foundation_tta_probe", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
            model.freeze_backbone()
        else:
            model = train_linear_probe(model, X_src, y_src, **self._probe_kwargs())
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        # Step 2: T3A — adjust head prototypes from unlabeled target predictions
        model = _run_foundation_t3a(
            model, target_unlabeled, self.device,
            n_classes=n_classes,
            confidence_threshold=self.t3a_confidence,
            batch_size=self.batch_size,
        )

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
            for start in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[start: start + self.batch_size]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)
