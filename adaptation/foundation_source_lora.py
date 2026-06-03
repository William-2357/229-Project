"""Source-finetuned LoRA adapters for pretrained foundation EEG backbones.

These adapters make the foundation LoRA path more comparable to specialist LoRA:
  1. start from a pretrained foundation encoder
  2. source-fine-tune backbone + head on pooled source subjects
  3. freeze the source-task-shaped backbone
  4. apply LoRA and adapt on target calibration data
"""

from __future__ import annotations

import copy
import time
import numpy as np
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model

from .base import BaseAdapter, train_epoch, evaluate_model
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_lora import _get_lora_target_modules
from .foundation_source_finetune import load_or_build_sft_model
from models.foundations import FoundationBackbone, FoundationWithHead


class FoundationSourceFineTuneLoRAAdapter(BaseAdapter):
    """Source-finetuned foundation backbone + LoRA target adaptation."""

    RANK_CANDIDATES = [4, 8, 16]

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        # Source fine-tuning params
        lr_src: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs_src: int = 200,
        patience_src: int = 25,
        val_fraction_src: float = 0.1,
        batch_size: int = 32,
        # LoRA params
        rank: int | None = None,
        lr_lora: float = 1e-3,
        max_epochs_lora: int = 100,
        patience_lora: int = 15,
        val_fraction_ft: float = 0.1,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationSourceFineTuneLoRAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed)
        self.lr_src = lr_src
        self.weight_decay = weight_decay
        self.max_epochs_src = max_epochs_src
        self.patience_src = patience_src
        self.val_fraction_src = val_fraction_src
        self.batch_size = batch_size
        self.rank = rank
        self.lr_lora = lr_lora
        self.max_epochs_lora = max_epochs_lora
        self.patience_lora = patience_lora
        self.val_fraction_ft = val_fraction_ft
        self._model: nn.Module | None = None
        self._selected_rank: int | None = None

    def _source_ft_kwargs(self) -> dict:
        return dict(
            device=self.device,
            lr_src=self.lr_src,
            weight_decay=self.weight_decay,
            max_epochs_src=self.max_epochs_src,
            patience_src=self.patience_src,
            val_fraction_src=self.val_fraction_src,
            batch_size=self.batch_size,
        )

    def _select_rank_by_source_cv(
        self, source_state: dict, n_classes: int, X_src: np.ndarray, y_src: np.ndarray
    ) -> int:
        n_val = max(1, int(len(X_src) * 0.2))
        idx = np.random.permutation(len(X_src))
        X_tr, y_tr = X_src[idx[n_val:]], y_src[idx[n_val:]]
        X_val, y_val = X_src[idx[:n_val]], y_src[idx[:n_val]]

        best_rank, best_acc = self.RANK_CANDIDATES[0], -1.0

        for r in self.RANK_CANDIDATES:
            model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
            model.load_state_dict(copy.deepcopy(source_state))
            model.freeze_backbone()

            target_modules = _get_lora_target_modules(model, rank=r)
            if not target_modules:
                continue

            lora_model = get_peft_model(model, LoraConfig(
                r=r,
                lora_alpha=r * 2,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
            ))

            optimizer = torch.optim.AdamW(
                [p for p in lora_model.parameters() if p.requires_grad],
                lr=self.lr_lora,
                weight_decay=self.weight_decay,
            )
            for _ in range(min(30, self.max_epochs_lora)):
                train_epoch(lora_model, X_tr, y_tr, optimizer, self.device, self.batch_size)

            acc = evaluate_model(lora_model, X_val, y_val, self.device)
            if acc > best_acc:
                best_acc = acc
                best_rank = r

        return best_rank

    def _finetune_lora(self, lora_model: nn.Module, X_cal: np.ndarray, y_cal: np.ndarray) -> nn.Module:
        if len(X_cal) < 2:
            return lora_model

        n_val = max(1, int(len(X_cal) * self.val_fraction_ft))
        idx = np.random.permutation(len(X_cal))
        val_idx, train_idx = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx

        X_tr, y_tr = X_cal[train_idx], y_cal[train_idx]
        X_val, y_val = X_cal[val_idx], y_cal[val_idx]

        optimizer = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad],
            lr=self.lr_lora,
            weight_decay=self.weight_decay,
        )
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
    ) -> "FoundationSourceFineTuneLoRAAdapter":
        if source_data is None:
            raise ValueError("FoundationSourceFineTuneLoRAAdapter requires source_data")

        self._seed()

        X_src, y_src = source_data
        n_classes = len(np.unique(y_src))

        # Source fine-tune + rank selection is a one-time, shared source-side cost
        # (excluded from fit_time — the timer below covers only per-K LoRA
        # adaptation). The fine-tune reuses a disk checkpoint shared across ALL SFT
        # methods/jobs/runs (see load_or_build_sft_model), so the ~3-4 min source
        # fine-tune isn't recomputed. Rank selection is cached in-memory per job.
        sft_model = load_or_build_sft_model(
            self.backbone, n_classes, X_src, y_src,
            seed=self.seed, source_cache=source_cache, **self._source_ft_kwargs(),
        )
        source_state = copy.deepcopy(sft_model.state_dict())
        if self.rank is not None:
            self._selected_rank = self.rank
        else:
            rank_key = ("foundation_sft_lora_rank", self.seed)
            if source_cache is not None and rank_key in source_cache:
                self._selected_rank = source_cache[rank_key]
            else:
                self._selected_rank = self._select_rank_by_source_cv(
                    source_state, n_classes, X_src, y_src)
                if source_cache is not None:
                    source_cache[rank_key] = self._selected_rank

        model = FoundationWithHead(copy.deepcopy(self.backbone), n_classes).to(self.device)
        model.load_state_dict(copy.deepcopy(source_state))
        model.freeze_backbone()

        # ---- fit_time covers LoRA adaptation only (frozen backbone is ready) ----
        t0 = time.time()
        if target_labeled is None or len(target_labeled[0]) < 2:
            self._model = model
            self._fit_time = time.time() - t0
            return self

        target_modules = _get_lora_target_modules(model, rank=self._selected_rank or 8)
        if not target_modules:
            self._model = model
            self._fit_time = time.time() - t0
            return self

        lora_model = get_peft_model(model, LoraConfig(
            r=self._selected_rank or 8,
            lora_alpha=(self._selected_rank or 8) * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
        ))

        X_cal, y_cal = target_labeled
        # train_time covers only the target LoRA epoch loop (excludes SFT load,
        # rank selection, and the LoRA model build).
        t_train = time.time()
        lora_model = self._finetune_lora(lora_model, X_cal, y_cal)
        self._train_time = time.time() - t_train

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


class FoundationSourceFineTuneEALoRAAdapter(FoundationSourceFineTuneLoRAAdapter):
    """EA whitening + source-finetuned foundation backbone + LoRA adaptation."""

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 epsilon: float = 1e-6, **kwargs):
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None,
            source_per_subject: list | None = None) -> "FoundationSourceFineTuneEALoRAAdapter":
        if source_data is None:
            raise ValueError("FoundationSourceFineTuneEALoRAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationSourceFineTuneEALoRAAdapter requires target_unlabeled for EA alignment")

        self._seed()
        X_src, y_src = source_data

        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(R_src))

        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        cal_aligned = None
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            cal_aligned = (euclidean_align(X_cal, self._target_R_inv_sqrt), y_cal)

        return super().fit(
            source_data=(X_src_aligned, y_src),
            target_unlabeled=euclidean_align(target_unlabeled, self._target_R_inv_sqrt),
            target_labeled=cal_aligned,
            source_cache=source_cache,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSourceFineTuneEALoRAAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationSourceFineTuneEALoRAAdapter not fitted")
        model = self._get_inference_model()
        model.eval()
        model.to(self.device)
        X_aligned = euclidean_align(X, self._target_R_inv_sqrt)
        probs = []
        with torch.no_grad():
            for start in range(0, len(X_aligned), self.batch_size):
                xb = torch.FloatTensor(X_aligned[start: start + self.batch_size]).to(self.device)
                probs.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
        return np.concatenate(probs)
