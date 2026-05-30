"""Dataset classes for BCIC-IV-2a EEG datasets."""

import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

from .preprocessing import preprocess_pipeline


class BaseEEGDataset(ABC):
    """Base class: subclasses must implement _load_raw_subject()."""

    def __init__(self, data_dir: str, target_sfreq: float = 200.0, epoch_len_sec: float = 4.0,
                 cache_dir: str | None = None):
        self.data_dir = Path(data_dir)
        self.target_sfreq = target_sfreq
        self.epoch_len_sec = epoch_len_sec
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict = {}

    @property
    @abstractmethod
    def subject_ids(self) -> list:
        """Return list of valid subject IDs."""

    @property
    @abstractmethod
    def n_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def orig_sfreq(self) -> float:
        pass

    @abstractmethod
    def _load_raw_subject(self, subject_id, session: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (raw_eeg, events, labels).

        raw_eeg: (C, T) float array at orig_sfreq
        events:  (N,) sample indices of cue onset
        labels:  (N,) integer class labels 0-indexed
        """

    def get_subject_data(
        self,
        subject_id,
        sessions: list[int] | None = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) preprocessed arrays for one subject.

        X: (N, C, T) float32
        y: (N,) int32
        """
        key = (subject_id, tuple(sessions or []))
        if use_cache and key in self._cache:
            return self._cache[key]

        sessions = sessions or self._default_sessions
        all_X, all_y = [], []
        for sess in sessions:
            cached = self._try_load_cache(subject_id, sess)
            if cached is not None:
                X, y = cached
            else:
                raw, events, labels = self._load_raw_subject(subject_id, sess)
                X, y = preprocess_pipeline(
                    raw=raw,
                    sfreq=self.orig_sfreq,
                    events=events,
                    labels=labels,
                    epoch_len_sec=self.epoch_len_sec,
                    tmin=0.0,
                    target_sfreq=self.target_sfreq,
                    notch_freqs=self._notch_freqs,
                )
                path = self._cache_path(subject_id, sess)
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, X=X, y=y)
            all_X.append(X)
            all_y.append(y)

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        if use_cache:
            self._cache[key] = (X, y)
        return X, y

    def _cache_path(self, subject_id, session: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"subj{subject_id:03d}_sess{session}.npz"

    def _try_load_cache(self, subject_id, session: int) -> tuple[np.ndarray, np.ndarray] | None:
        path = self._cache_path(subject_id, session)
        if path is None or not path.exists():
            return None
        try:
            d = np.load(path)
            return d["X"], d["y"]
        except Exception:
            print(f"  [cache] corrupted {path.name}, falling back to raw preprocessing", flush=True)
            return None

    def save_preprocessed_cache(self, cache_dir: str) -> None:
        """Preprocess all subjects/sessions and save to cache_dir. Run once before training."""
        out = Path(cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        for subject_id in self.subject_ids:
            for sess in self._default_sessions:
                path = out / f"subj{subject_id:03d}_sess{sess}.npz"
                if path.exists():
                    try:
                        d = np.load(path)
                        _ = d["X"], d["y"]
                        print(f"  skip {path.name} (exists, valid)")
                        continue
                    except Exception:
                        print(f"  overwrite {path.name} (corrupted)")
                print(f"  processing subj={subject_id} sess={sess} ...", flush=True)
                raw, events, labels = self._load_raw_subject(subject_id, sess)
                X, y = preprocess_pipeline(
                    raw=raw,
                    sfreq=self.orig_sfreq,
                    events=events,
                    labels=labels,
                    epoch_len_sec=self.epoch_len_sec,
                    tmin=0.0,
                    target_sfreq=self.target_sfreq,
                    notch_freqs=self._notch_freqs,
                )
                np.savez_compressed(path, X=X, y=y)
                print(f"    saved {path.name}  X={X.shape}")


class BCICIVDataset(BaseEEGDataset):
    """BCI Competition IV Dataset 2a.

    9 subjects, 4-class motor imagery, 22-channel EEG at 250 Hz, 2 sessions.
    Expected directory layout:
        data_dir/
            A01T.npz  (session 1, training)
            A01E.npz  (session 2, evaluation)
            ...
    """

    N_SUBJECTS = 9
    N_CLASSES = 4
    N_CHANNELS = 22
    ORIG_SFREQ = 250.0
    _notch_freqs = (50.0,)
    _default_sessions = [1]

    # Class labels: 1=left hand, 2=right hand, 3=feet, 4=tongue → remapped to 0-3
    CLASS_REMAP = {1: 0, 2: 1, 3: 2, 4: 3}

    @property
    def subject_ids(self) -> list:
        return list(range(1, self.N_SUBJECTS + 1))

    @property
    def n_classes(self) -> int:
        return self.N_CLASSES

    @property
    def n_channels(self) -> int:
        return self.N_CHANNELS

    @property
    def orig_sfreq(self) -> float:
        return self.ORIG_SFREQ

    def _load_raw_subject(self, subject_id: int, session: int) -> tuple:
        suffix = "T" if session == 1 else "E"
        npz_path = self.data_dir / f"A{subject_id:02d}{suffix}.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {npz_path}\n"
                "Run: python -m data.download --dataset bciciv2a --data_dir <dir>"
            )

        d = np.load(npz_path, allow_pickle=True)
        raw = d["eeg"].astype(np.float64)   # (C, T)
        events = d["events"].astype(int)
        labels = np.array([self.CLASS_REMAP[int(l)] for l in d["labels"]], dtype=int)
        return raw, events, labels

    def get_source_data(self, held_out_subject: int) -> tuple[np.ndarray, np.ndarray]:
        all_X, all_y = [], []
        for subj in self.subject_ids:
            if subj == held_out_subject:
                continue
            X, y = self.get_subject_data(subj, sessions=[1])
            all_X.append(X)
            all_y.append(y)
        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    def get_target_data(self, subject_id: int) -> tuple[tuple, tuple]:
        # E files have no labels (competition holdout) — split T session into 80/20
        X, y = self.get_subject_data(subject_id, sessions=[1])
        split = int(len(X) * 0.8)
        return (X[:split], y[:split]), (X[split:], y[split:])
