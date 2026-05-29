"""LoRA fine-tuning adapter using the peft library.

Applies low-rank adapters to Conv2d/Linear layers of the backbone.
Backbone is frozen; only LoRA parameters are trained on the calibration set.

Rank ablation: r in {4, 8, 16}, selected by source-subject CV only (never target).
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from .base import BaseAdapter, train_epoch, evaluate_model
from models.specialists import get_conv_layer_names


def _get_lora_target_modules(model: nn.Module, rank: int = 4) -> list[str]:
    """Get Conv2d and Linear layer names for LoRA targeting.

    Excludes grouped convolutions (depthwise) where rank is not divisible by
    groups, which peft does not support.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            targets.append(name)
        elif isinstance(module, nn.Conv2d):
            # peft requires rank % groups == 0
            if module.groups == 1 or rank % module.groups == 0:
                targets.append(name)
    return targets


def build_lora_model(model: nn.Module, rank: int, target_modules: list[str]) -> nn.Module:
    """Wrap model with LoRA adapters on specified layers.

    Uses peft.get_peft_model. Note: modifies model in-place.
    lora_alpha = 2 * rank (common heuristic for stable scaling).
    """
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        # task_type not set: custom PyTorch model (not HuggingFace PreTrainedModel)
    )
    return get_peft_model(model, config)


class LoRAAdapter(BaseAdapter):
    """LoRA fine-tuning on target calibration data.

    Workflow:
        1. Pre-train backbone on source subjects (LOSO)
        2. Wrap with LoRA adapters (freeze backbone, train only adapters)
        3. Fine-tune LoRA params on target calibration set
    """

    RANK_CANDIDATES = [4, 8, 16]

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 # Source pre-training
                 lr_src: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs_src: int = 200, batch_size: int = 64,
                 patience_src: int = 20, val_fraction_src: float = 0.1,
                 # LoRA fine-tuning
                 rank: int | None = None,  # None = auto-select by source CV
                 lr_lora: float = 1e-3, max_epochs_lora: int = 100,
                 patience_lora: int = 15, val_fraction_ft: float = 0.1):
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.batch_size = batch_size
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.rank = rank
        self.lr_lora = lr_lora
        self.max_epochs_lora = max_epochs_lora
        self.patience_lora = patience_lora
        self.val_fraction_ft = val_fraction_ft
        self._model: nn.Module | None = None
        self._selected_rank: int | None = None

    def _train_source(self, model: nn.Module, X_src: np.ndarray, y_src: np.ndarray) -> nn.Module:
        n_val = max(1, int(len(X_src) * self.val_fraction_src))
        idx = np.random.permutation(len(X_src))
        X_tr, y_tr = X_src[idx[n_val:]], y_src[idx[n_val:]]
        X_val, y_val = X_src[idx[:n_val]], y_src[idx[:n_val]]

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

    def _select_rank_by_source_cv(
        self, model_state: dict, X_src: np.ndarray, y_src: np.ndarray, target_modules: list[str]
    ) -> int:
        """Select best LoRA rank using source-subject held-out validation.

        Splits source data 80/20, evaluates each rank candidate, returns best.
        """
        from sklearn.model_selection import train_test_split

        n_val = max(1, int(len(X_src) * 0.2))
        idx = np.random.permutation(len(X_src))
        X_tr, y_tr = X_src[idx[n_val:]], y_src[idx[n_val:]]
        X_val, y_val = X_src[idx[:n_val]], y_src[idx[:n_val]]

        best_rank, best_acc = self.RANK_CANDIDATES[0], -1.0

        for r in self.RANK_CANDIDATES:
            # Fresh copy for each rank
            backbone_copy = copy.deepcopy(self.backbone).to(self.device)
            backbone_copy.load_state_dict(model_state)

            # Re-filter target modules for this specific rank
            rank_modules = _get_lora_target_modules(backbone_copy, rank=r)
            if not rank_modules:
                continue

            try:
                lora_model = build_lora_model(backbone_copy, rank=r, target_modules=rank_modules)
            except Exception:
                continue

            # Quick fine-tune on source train split
            lora_optimizer = torch.optim.AdamW(
                [p for p in lora_model.parameters() if p.requires_grad],
                lr=self.lr_lora,
            )
            for _ in range(min(30, self.max_epochs_lora)):
                train_epoch(lora_model, X_tr, y_tr, lora_optimizer, self.device, self.batch_size)

            acc = evaluate_model(lora_model, X_val, y_val, self.device)
            if acc > best_acc:
                best_acc = acc
                best_rank = r

        return best_rank

    def _finetune_lora(
        self, lora_model: nn.Module, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> nn.Module:
        if len(X_cal) < 2:
            return lora_model

        n_val = max(1, int(len(X_cal) * self.val_fraction_ft))
        idx = np.random.permutation(len(X_cal))
        if len(idx) <= n_val:
            train_idx, val_idx = idx, idx
        else:
            val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]

        trainable_params = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr_lora, weight_decay=self.weight_decay)
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

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None) -> "LoRAAdapter":
        if source_data is None:
            raise ValueError("LoRAAdapter requires source_data")
        if target_labeled is None:
            # k=0: no calibration data — train source model only, skip LoRA
            self._seed()
            t0 = time.time()
            X_src, y_src = source_data
            cache_key = self.seed
            model = self._clone_backbone().to(self.device)
            if source_cache is not None and cache_key in source_cache:
                cached = source_cache[cache_key]
                model.load_state_dict(copy.deepcopy(cached["state_dict"]))
                self._selected_rank = cached["rank"]
            else:
                model = self._train_source(model, X_src, y_src)
                source_state = copy.deepcopy(model.state_dict())
                target_modules = _get_lora_target_modules(model, rank=min(self.RANK_CANDIDATES))
                if target_modules:
                    self._selected_rank = (self.rank if self.rank is not None
                                           else self._select_rank_by_source_cv(
                                               source_state, X_src, y_src, target_modules))
                else:
                    self._selected_rank = min(self.RANK_CANDIDATES)
                if source_cache is not None:
                    source_cache[cache_key] = {"state_dict": source_state, "rank": self._selected_rank}
            self._model = model
            self._fit_time = time.time() - t0
            return self

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        X_cal, y_cal = target_labeled

        # Step 1: Source pre-training (cached by seed to avoid re-training across K values)
        # Original (no caching):
        # model = self._clone_backbone().to(self.device)
        # model = self._train_source(model, X_src, y_src)
        # source_state = copy.deepcopy(model.state_dict())
        # ... then steps 2-3 below ...
        cache_key = self.seed
        if source_cache is not None and cache_key in source_cache:
            cached = source_cache[cache_key]
            source_state = cached["state_dict"]
            self._selected_rank = cached["rank"]
        else:
            model = self._clone_backbone().to(self.device)
            model = self._train_source(model, X_src, y_src)
            source_state = copy.deepcopy(model.state_dict())

            # Step 2: Discover target modules (rank known after selection below, use min candidate)
            # We first discover with the minimum rank to filter incompatible layers
            target_modules = _get_lora_target_modules(model, rank=min(self.RANK_CANDIDATES))
            if not target_modules:
                # Fallback: no LoRA-compatible layers → full fine-tune
                from .finetune import FineTuneAdapter
                ft = FineTuneAdapter(self.backbone, str(self.device), self.seed)
                ft.fit(source_data, target_unlabeled, target_labeled)
                self._model = ft._model
                self._fit_time = time.time() - t0
                return self

            # Step 3: Select rank (source CV only, never target)
            if self.rank is not None:
                self._selected_rank = self.rank
            else:
                self._selected_rank = self._select_rank_by_source_cv(
                    source_state, X_src, y_src, target_modules
                )

            if source_cache is not None:
                source_cache[cache_key] = {"state_dict": source_state, "rank": self._selected_rank}

        # Step 4: Build LoRA model and fine-tune on target calibration
        # Fresh backbone with source weights; re-filter target_modules for selected rank
        backbone_for_lora = copy.deepcopy(self.backbone).to(self.device)
        backbone_for_lora.load_state_dict(copy.deepcopy(source_state))
        final_target_modules = _get_lora_target_modules(backbone_for_lora, rank=self._selected_rank)

        lora_model = build_lora_model(backbone_for_lora, self._selected_rank, final_target_modules)
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
            for start in range(0, len(X), 64):
                xb = torch.FloatTensor(X[start: start + 64]).to(self.device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)
