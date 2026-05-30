"""LoRA fine-tuning adapter for pretrained foundation EEG backbones.

Unlike LoRAAdapter (which trains the backbone from scratch on source data),
this adapter skips source pre-training and applies LoRA directly to the
pretrained foundation backbone.

For k=0:  trains only the linear classification head on source features
          (linear probe — backbone stays fully frozen).
For k>0:  applies LoRA to the frozen backbone's Linear layers and fine-tunes
          LoRA params + head on the calibration set.

Rank selection: without source data, CV-based rank selection is unreliable at
small K. A fixed default rank (8) is used; override with rank= if needed.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from .base import BaseAdapter, train_epoch, evaluate_model
from models.foundations import FoundationBackbone, FoundationWithHead


class _HeadWrapper(nn.Module):
    """Thin nn.Module that applies a linear head to pre-extracted feature vectors."""
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _get_linear_target_modules(model: nn.Module, rank: int = 8) -> list[str]:
    """Return names of nn.Linear layers eligible for LoRA.

    Skips the classification head (named 'head') — only the backbone's
    attention and feed-forward layers should receive LoRA adapters.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.endswith("head"):
            targets.append(name)
    return targets


class FoundationLoRAAdapter(BaseAdapter):
    """LoRA fine-tuning of a pretrained foundation backbone + linear head.

    Workflow:
        k=0: freeze backbone, train linear head on source features (linear probe)
        k>0: apply LoRA to frozen backbone Linear layers,
             fine-tune LoRA params + head on calibration set
    """

    RANK_CANDIDATES = [4, 8, 16]

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        # Linear probe (k=0) params
        lr_probe: float = 1e-3,
        max_epochs_probe: int = 100,
        patience_probe: int = 15,
        val_fraction_probe: float = 0.1,
        # LoRA fine-tune (k>0) params
        rank: int = 8,              # fixed default; no source CV available
        lr_lora: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_lora: int = 100,
        patience_lora: int = 15,
        val_fraction_ft: float = 0.1,
        batch_size: int = 32,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationLoRAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_probe = lr_probe
        self.max_epochs_probe = max_epochs_probe
        self.patience_probe = patience_probe
        self.val_fraction_probe = val_fraction_probe
        self.rank = rank
        self.lr_lora = lr_lora
        self.weight_decay = weight_decay
        self.max_epochs_lora = max_epochs_lora
        self.patience_lora = patience_lora
        self.val_fraction_ft = val_fraction_ft
        self.batch_size = batch_size
        self._model: nn.Module | None = None

    def _train_linear_probe(
        self, model: FoundationWithHead, X: np.ndarray, y: np.ndarray
    ) -> FoundationWithHead:
        """Train only the classification head with backbone frozen.

        Pre-extracts backbone features once so the epoch loop runs only the
        linear head — avoids 100x redundant forward passes through the large
        frozen backbone.
        """
        model.freeze_backbone()

        # Extract features once (backbone frozen, so features are constant)
        model.eval()
        feats = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                xb = torch.FloatTensor(X[start: start + self.batch_size]).to(self.device)
                feats.append(model.backbone.get_features(xb).cpu().numpy())
        X_feat = np.concatenate(feats, axis=0)

        n_val = max(1, int(len(X_feat) * self.val_fraction_probe))
        idx = np.random.permutation(len(X_feat))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_feat[train_idx], y[train_idx]
        X_val, y_val = X_feat[val_idx], y[val_idx]

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
        return model

    def _finetune_lora(
        self, lora_model: nn.Module, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> nn.Module:
        """Fine-tune LoRA params + head on calibration data."""
        if len(X_cal) < 2:
            return lora_model

        n_val = max(1, int(len(X_cal) * self.val_fraction_ft))
        idx = np.random.permutation(len(X_cal))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]

        trainable = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.lr_lora, weight_decay=self.weight_decay)
        best_val_acc, best_state, patience_counter = -1.0, None, 0

        for _ in range(self.max_epochs_lora):
            train_epoch(lora_model, X_tr, y_tr, optimizer, self.device, self.batch_size)
            val_acc = evaluate_model(lora_model, X_val, y_val, self.device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(lora_model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience_lora:
                break

        if best_state is not None:
            lora_model.load_state_dict(best_state)
        return lora_model

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
    ) -> "FoundationLoRAAdapter":
        if source_data is None:
            raise ValueError("FoundationLoRAAdapter requires source_data")

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Build backbone + head; backbone starts frozen
        model = FoundationWithHead(
            copy.deepcopy(self.backbone), n_classes
        ).to(self.device)

        # Step 1: Linear probe on source features (warms up the head)
        # Probe result is identical across all K/repeat calls (same X_src + seed) — cache it.
        cache_key = ("foundation_lora_probe", self.seed)
        if source_cache is not None and cache_key in source_cache:
            model.load_state_dict(copy.deepcopy(source_cache[cache_key]))
        else:
            model = self._train_linear_probe(model, X_src, y_src)
            if source_cache is not None:
                source_cache[cache_key] = copy.deepcopy(model.state_dict())

        if target_labeled is None or len(target_labeled[0]) < 2:
            # k=0: linear probe only — no LoRA adaptation
            self._model = model
            self._fit_time = time.time() - t0
            return self

        # Step 2: Apply LoRA to backbone Linear layers (backbone stays frozen)
        target_modules = _get_linear_target_modules(model, rank=self.rank)
        if not target_modules:
            # No eligible Linear layers — fall back to linear probe result
            self._model = model
            self._fit_time = time.time() - t0
            return self

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        )
        # get_peft_model marks LoRA params as trainable; backbone weights stay frozen
        lora_model = get_peft_model(model, lora_config)

        # Step 3: Fine-tune LoRA params + head on target calibration
        X_cal, y_cal = target_labeled
        lora_model = self._finetune_lora(lora_model, X_cal, y_cal)

        self._model = lora_model
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
