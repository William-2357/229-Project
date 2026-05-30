"""Synthetic EEG dataset for smoke testing without downloading real data.

Generates reproducible per-subject data with mild class-discriminative
structure so adaptation methods have something real to do.
"""

import numpy as np
from .datasets import BaseEEGDataset


class SyntheticDataset(BaseEEGDataset):
    """Drop-in replacement for BCICIVDataset.

    Generates (C, T) arrays that look like preprocessed EEG:
    - Each class gets a small sinusoidal bump at a class-specific frequency
      added on top of pink noise, so classifiers can actually learn.
    - Each subject has slightly different noise variance and class means
      (controlled by subject_id seed) to simulate realistic cross-subject variability.

    No files needed; all data generated in memory.

    Args:
        n_subjects:   Number of fake subjects (default 5)
        n_classes:    Number of classes (default 4)
        n_channels:   EEG channels (default 22)
        trials_per_class_per_session: Trials per class per session (default 20)
        n_sessions:   Number of sessions (default 3; session 3 = held-out test)
        target_sfreq: Already-resampled frequency — data is generated at this rate
        epoch_len_sec: Epoch length in seconds
        seed:         Global RNG seed
    """

    def __init__(
        self,
        data_dir: str = "",           # ignored — no files needed
        n_subjects: int = 5,
        n_classes: int = 4,
        n_channels: int = 22,
        trials_per_class_per_session: int = 20,
        n_sessions: int = 3,
        target_sfreq: float = 200.0,
        epoch_len_sec: float = 4.0,
        seed: int = 0,
    ):
        super().__init__(data_dir, target_sfreq=target_sfreq, epoch_len_sec=epoch_len_sec)
        self._n_subjects = n_subjects
        self._n_classes = n_classes
        self._n_channels = n_channels
        self._trials_per_class = trials_per_class_per_session
        self._n_sessions = n_sessions
        self._seed = seed
        self._default_sessions = list(range(1, n_sessions))  # all but last = train

        # Pre-generate all subject data upfront for speed
        self._data: dict = {}
        self._generate_all()

    # ------------------------------------------------------------------
    # BaseEEGDataset interface
    # ------------------------------------------------------------------

    @property
    def subject_ids(self) -> list:
        return list(range(1, self._n_subjects + 1))

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def orig_sfreq(self) -> float:
        return self.target_sfreq  # already at target rate

    def _load_raw_subject(self, subject_id, session):
        # Not called — we override get_subject_data directly
        raise NotImplementedError("SyntheticDataset overrides get_subject_data directly")

    # ------------------------------------------------------------------
    # Override get_subject_data to skip file I/O and preprocessing
    # ------------------------------------------------------------------

    def get_subject_data(
        self,
        subject_id,
        sessions=None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        sessions = sessions or self._default_sessions
        key = (subject_id, tuple(sessions))
        if use_cache and key in self._cache:
            return self._cache[key]
        all_X, all_y = [], []
        for sess in sessions:
            X, y = self._data[(subject_id, sess)]
            all_X.append(X)
            all_y.append(y)
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        if use_cache:
            self._cache[key] = (X, y)
        return X, y

    def get_source_data(self, held_out_subject: int) -> tuple[np.ndarray, np.ndarray]:
        all_X, all_y = [], []
        train_sessions = list(range(1, self._n_sessions))  # sessions 1..n-1
        for subj in self.subject_ids:
            if subj == held_out_subject:
                continue
            X, y = self.get_subject_data(subj, sessions=train_sessions)
            all_X.append(X)
            all_y.append(y)
        return np.concatenate(all_X), np.concatenate(all_y)

    def get_target_data(self, subject_id: int) -> tuple[tuple, tuple]:
        train_sessions = list(range(1, self._n_sessions))
        test_session = [self._n_sessions]
        X_tr, y_tr = self.get_subject_data(subject_id, sessions=train_sessions)
        X_te, y_te = self.get_subject_data(subject_id, sessions=test_session)
        return (X_tr, y_tr), (X_te, y_te)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _generate_all(self) -> None:
        n_t = int(self.epoch_len_sec * self.target_sfreq)
        t = np.linspace(0, self.epoch_len_sec, n_t, endpoint=False)

        # Class frequencies FIXED across subjects so cross-subject transfer is possible
        # (mimics real EEG where motor imagery reliably modulates mu/beta bands)
        class_freqs = [10.0, 12.0, 20.0, 24.0]  # mu and beta band, 4 classes
        # Fixed spatial pattern per class (same channels across subjects, small jitter)
        class_channels = [
            [0, 1, 2, 3],    # class 0: frontal
            [5, 6, 7, 8],    # class 1: central-left
            [10, 11, 12, 13], # class 2: central-right
            [18, 19, 20, 21], # class 3: parietal
        ]

        for subj in self.subject_ids:
            rng = np.random.default_rng(self._seed + subj * 1000)
            # Per-subject variability: only noise scale and SNR vary, not class structure
            noise_scale = rng.uniform(0.8, 1.2)
            snr = rng.uniform(0.15, 0.25)

            for sess in range(1, self._n_sessions + 1):
                sess_rng = np.random.default_rng(self._seed + subj * 1000 + sess * 100)
                X_list, y_list = [], []
                for cls in range(self._n_classes):
                    for trial_idx in range(self._trials_per_class):
                        # Pink noise background: (C, T)
                        white = sess_rng.standard_normal((self._n_channels, n_t))
                        pink = np.cumsum(white, axis=-1)
                        pink /= (np.std(pink, axis=-1, keepdims=True) + 1e-8)
                        pink *= noise_scale

                        # Class signal: fixed frequency + fixed channels, only amplitude/phase vary
                        freq = class_freqs[cls % len(class_freqs)]
                        freq += sess_rng.uniform(-0.5, 0.5)  # small jitter only
                        channels = class_channels[cls % len(class_channels)]

                        amplitude = sess_rng.uniform(0.5, 1.0)
                        phase = sess_rng.uniform(0, 2 * np.pi)
                        sine = amplitude * np.sin(2 * np.pi * freq * t + phase)

                        trial = pink.copy()
                        trial[channels] += snr * sine

                        # Per-trial z-score (matches preprocessing)
                        mean = trial.mean(axis=-1, keepdims=True)
                        std = trial.std(axis=-1, keepdims=True) + 1e-8
                        trial = ((trial - mean) / std).astype(np.float32)

                        X_list.append(trial)
                        y_list.append(cls)

                # Shuffle within session
                idx = sess_rng.permutation(len(X_list))
                X = np.stack(X_list)[idx]
                y = np.array(y_list, dtype=np.int32)[idx]
                self._data[(subj, sess)] = (X, y)
