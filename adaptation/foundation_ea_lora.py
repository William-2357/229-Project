"""EA + LoRA adapter for pretrained foundation EEG backbones.

Mirrors EALoRAAdapter but skips backbone source-training.  Applies Euclidean
Alignment to raw EEG before passing through the frozen foundation backbone,
then fine-tunes LoRA adapters + linear head on the aligned calibration set.

For k=0: EA alignment + linear probe on aligned source (no LoRA).
For k>0: EA alignment + LoRA fine-tuning on aligned calibration set.
"""

import copy
import numpy as np
import torch.nn as nn

from .ea import compute_mean_covariance, matrix_sqrt_inv, euclidean_align
from .foundation_lora import FoundationLoRAAdapter
from models.foundations import FoundationBackbone


class FoundationEALoRAAdapter(FoundationLoRAAdapter):
    """EA whitening + LoRA fine-tuning on a frozen pretrained foundation backbone.

    Workflow:
        1. Compute EA transform from unlabeled target trials
        2. Align source data (per-subject if source_per_subject provided)
        3. Align calibration data with the target EA transform
        4. Run FoundationLoRAAdapter on the aligned data
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: str = "cpu",
        seed: int = 42,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        if not isinstance(backbone, FoundationBackbone):
            raise TypeError(
                f"FoundationEALoRAAdapter requires a FoundationBackbone, "
                f"got {type(backbone).__name__}."
            )
        super().__init__(backbone, device, seed, **kwargs)
        self.epsilon = epsilon
        self._target_R_inv_sqrt: np.ndarray | None = None

    def fit(
        self, source_data, target_unlabeled=None, target_labeled=None,
        source_cache: dict | None = None,
        source_per_subject: list | None = None,
    ) -> "FoundationEALoRAAdapter":
        if source_data is None:
            raise ValueError("FoundationEALoRAAdapter requires source_data")
        if target_unlabeled is None:
            raise ValueError("FoundationEALoRAAdapter requires target_unlabeled for EA alignment")

        self._seed()

        X_src, y_src = source_data

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

        # Step 2: Compute target EA transform from unlabeled target trials
        R_tgt = compute_mean_covariance(target_unlabeled, self.epsilon)
        self._target_R_inv_sqrt = matrix_sqrt_inv(R_tgt)

        # Step 3: Align calibration data if present
        cal_aligned = None
        if target_labeled is not None:
            X_cal, y_cal = target_labeled
            cal_aligned = (euclidean_align(X_cal, self._target_R_inv_sqrt), y_cal)

        # Step 4: LoRA fit on aligned data
        return super().fit(
            source_data=(X_src_aligned, y_src),
            target_unlabeled=euclidean_align(target_unlabeled, self._target_R_inv_sqrt),
            target_labeled=cal_aligned,
            source_cache=source_cache,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEALoRAAdapter not fitted")
        return super().predict(euclidean_align(X, self._target_R_inv_sqrt))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._target_R_inv_sqrt is None:
            raise RuntimeError("FoundationEALoRAAdapter not fitted")
        return super().predict_proba(euclidean_align(X, self._target_R_inv_sqrt))
