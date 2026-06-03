"""EA + LoRA stacked adapter.

Applies Euclidean Alignment preprocessing then LoRA fine-tuning.
Tests whether unsupervised alignment and supervised PEFT are complementary.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter
from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .lora import LoRAAdapter


class EALoRAAdapter(BaseAdapter):
    """EA preprocessing + LoRA fine-tuning (stacked).

    Workflow:
        1. Compute EA transform from unlabeled target trials
        2. Apply EA to all source data and target calibration data
        3. Run LoRAAdapter on aligned data
    """

    def __init__(self, backbone: nn.Module, device: str = "cpu", seed: int = 42,
                 # EA params
                 epsilon: float = 1e-6,
                 # LoRA params
                 lr_src: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs_src: int = 200, batch_size: int = 64,
                 patience_src: int = 20, val_fraction_src: float = 0.1,
                 rank: int | None = None,
                 lr_lora: float = 1e-3, max_epochs_lora: int = 100,
                 patience_lora: int = 15, val_fraction_ft: float = 0.1):
        super().__init__(backbone, device, seed)
        self.epsilon = epsilon
        self._lora_kwargs = dict(
            lr_src=lr_src, weight_decay=weight_decay,
            max_epochs_src=max_epochs_src, batch_size=batch_size,
            patience_src=patience_src, val_fraction_src=val_fraction_src,
            rank=rank, lr_lora=lr_lora, max_epochs_lora=max_epochs_lora,
            patience_lora=patience_lora, val_fraction_ft=val_fraction_ft,
        )
        self._lora_adapter: LoRAAdapter | None = None
        self._target_R_inv_sqrt: np.ndarray | None = None
        self._source_R_inv_sqrt: np.ndarray | None = None

    def fit(self, source_data, target_unlabeled=None, target_labeled=None,
            source_cache: dict | None = None,
            source_per_subject: list | None = None) -> "EALoRAAdapter":
        if source_data is None:
            raise ValueError("EALoRAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("EALoRAAdapter requires target_unlabeled for EA")
        if target_labeled is None:
            # k=0: no calibration data — apply EA to source then train source-only model
            self._seed()
            t0 = time.time()
            X_src, y_src = source_data
            R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
            self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)
            if source_per_subject is not None:
                aligned_chunks = []
                for X_subj, _ in source_per_subject:
                    R = compute_mean_covariance(X_subj, self.epsilon)
                    aligned_chunks.append(euclidean_align(X_subj, matrix_sqrt_inv(R)))
                X_src_aligned = np.concatenate(aligned_chunks, axis=0)
            else:
                X_src_aligned = euclidean_align(X_src, matrix_sqrt_inv(
                    compute_mean_covariance(X_src, self.epsilon)))
            self._lora_adapter = LoRAAdapter(
                backbone=copy.deepcopy(self.backbone), device=str(self.device), seed=self.seed,
                **self._lora_kwargs,
            )
            self._lora_adapter.fit(source_data=(X_src_aligned, y_src),
                                   target_labeled=None, source_cache=source_cache)
            self._train_time = self._lora_adapter.train_time
            self._fit_time = time.time() - t0
            return self

        self._seed()
        t0 = time.time()

        X_src, y_src = source_data
        X_cal, y_cal = target_labeled

        # Step 1: Compute target EA whitening matrix
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Step 2: Align source per-subject independently (He & Wu 2020)
        if source_per_subject is not None:
            aligned_chunks = []
            for X_subj, _ in source_per_subject:
                R = compute_mean_covariance(X_subj, self.epsilon)
                R_inv_sqrt = matrix_sqrt_inv(R)
                aligned_chunks.append(euclidean_align(X_subj, R_inv_sqrt))
            X_src_aligned = np.concatenate(aligned_chunks, axis=0)
        else:
            R_src = compute_mean_covariance(X_src, self.epsilon)
            self._source_R_inv_sqrt = matrix_sqrt_inv(R_src)
            X_src_aligned = euclidean_align(X_src, self._source_R_inv_sqrt)
        X_cal_aligned = euclidean_align(X_cal, self._target_R_inv_sqrt)
        X_unlabeled_aligned = euclidean_align(target_unlabeled, self._target_R_inv_sqrt)

        # Step 3: LoRA on aligned data
        self._lora_adapter = LoRAAdapter(
            backbone=self.backbone,
            device=str(self.device),
            seed=self.seed,
            **self._lora_kwargs,
        )
        self._lora_adapter.fit(
            source_data=(X_src_aligned, y_src),
            target_unlabeled=X_unlabeled_aligned,
            target_labeled=(X_cal_aligned, y_cal),
            source_cache=source_cache,
        )

        self._train_time = self._lora_adapter.train_time
        self._fit_time = time.time() - t0
        return self

    def _align_target(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            return X
        return euclidean_align(X, self._target_R_inv_sqrt)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aligned = self._align_target(X)
        return self._lora_adapter.predict(X_aligned)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aligned = self._align_target(X)
        return self._lora_adapter.predict_proba(X_aligned)
