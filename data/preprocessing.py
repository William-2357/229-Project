"""EEG preprocessing pipeline: CAR, bandpass, notch, resample, epoch, z-score."""

import numpy as np
import mne.filter
from scipy.signal import resample_poly


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Apply CAR to (C, T) or (N, C, T) array."""
    return data - data.mean(axis=-2, keepdims=True)


def bandpass_filter(data: np.ndarray, sfreq: float, l_freq: float = 4.0, h_freq: float = 40.0) -> np.ndarray:
    """Zero-phase FIR bandpass on (..., T) array."""
    return mne.filter.filter_data(
        data.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        copy=True,
        verbose=False,
    )


def notch_filter(data: np.ndarray, sfreq: float, freqs=(50.0, 60.0)) -> np.ndarray:
    """Notch filter at specified frequencies on (..., T) array.
    Note: mne.filter.notch_filter uses 'Fs', not 'sfreq'.
    """
    notch_freqs = [f for f in freqs if f < sfreq / 2]
    if not notch_freqs:
        return data
    return mne.filter.notch_filter(
        data.astype(np.float64),
        Fs=sfreq,
        freqs=np.array(notch_freqs),
        method="fir",
        fir_design="firwin",
        phase="zero",
        copy=True,
        verbose=False,
    )


def resample(data: np.ndarray, orig_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Polyphase resample (..., T) array from orig_sfreq to target_sfreq."""
    if orig_sfreq == target_sfreq:
        return data
    from math import gcd
    ratio = int(orig_sfreq), int(target_sfreq)
    g = gcd(*ratio)
    down, up = ratio[0] // g, ratio[1] // g
    return resample_poly(data, up=up, down=down, axis=-1).astype(np.float32)


def extract_epochs(
    raw: np.ndarray,
    events: np.ndarray,
    sfreq: float,
    tmin: float = 0.0,
    epoch_len_sec: float = 4.0,
) -> np.ndarray:
    """Extract fixed-length epochs from a continuous (C, T) recording.

    Args:
        raw: (C, T) raw EEG array
        events: (N,) array of sample indices for cue onset
        sfreq: sampling frequency in Hz
        tmin: start offset relative to cue in seconds
        epoch_len_sec: epoch duration in seconds

    Returns:
        (N, C, T_epoch) float32 array
    """
    n_samples = int(epoch_len_sec * sfreq)
    start_offset = int(tmin * sfreq)
    epochs = []
    for onset in events:
        start = onset + start_offset
        end = start + n_samples
        if start < 0 or end > raw.shape[-1]:
            continue
        epochs.append(raw[:, start:end])
    return np.array(epochs, dtype=np.float32)


def zscore_per_trial(X: np.ndarray) -> np.ndarray:
    """Per-channel, per-trial z-score. Input: (N, C, T), output: same shape."""
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mean) / std


def preprocess_pipeline(
    raw: np.ndarray,
    sfreq: float,
    events: np.ndarray,
    labels: np.ndarray,
    epoch_len_sec: float = 4.0,
    tmin: float = 0.0,
    target_sfreq: float = 200.0,
    notch_freqs: tuple = (50.0, 60.0),
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    apply_zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline returning (X, y).

    Steps:
        1. Common-average reference
        2. Bandpass (default 4–40 Hz for specialist models; use 0.1–75 Hz for LaBraM)
        3. Notch 50/60 Hz
        4. Resample to target_sfreq
        5. Epoch (4 s aligned to cue)
        6. Per-channel, per-trial z-score (skipped for LaBraM — µV scale preserved)

    Args:
        l_freq: bandpass lower cutoff in Hz
        h_freq: bandpass upper cutoff in Hz
        apply_zscore: if False, skip z-scoring (use for LaBraM which normalises via ÷100 in µV)

    Returns:
        X: (N, C, T) float32
        y: (N,) int32
    """
    data = common_average_reference(raw)
    data = bandpass_filter(data, sfreq, l_freq=l_freq, h_freq=h_freq)
    data = notch_filter(data, sfreq, freqs=notch_freqs)
    data = resample(data, sfreq, target_sfreq)

    # Rescale event indices to new sampling rate
    scale = target_sfreq / sfreq
    scaled_events = (events * scale).astype(int)

    X = extract_epochs(data, scaled_events, target_sfreq, tmin=tmin, epoch_len_sec=epoch_len_sec)

    # Keep only valid epochs (events that fit in recording)
    n_valid = len(X)
    y = labels[:n_valid].astype(np.int32)

    if apply_zscore:
        X = zscore_per_trial(X)
    return X, y


# ---------------------------------------------------------------------------
# Per-backbone preprocessing configs
#
# Each foundation model was pretrained on a specific bandpass and normalisation
# scheme. Mismatching the bandpass at inference degrades pretrained features.
#
# IMPORTANT — cache directories must differ per config (the cache stores the
# preprocessed arrays; mixing configs in the same directory silently corrupts
# results).
# ---------------------------------------------------------------------------

# LaBraM (Jiang et al. 2024) — pretrained on TUAB and BCIC data.
# Bandpass: 0.1–75 Hz (per the official preprocessing in the paper).
# Normalisation: raw Volts preserved here; LaBraMBackbone.get_features()
# converts V→µV then divides by 100 µV before the transformer.
# apply_zscore=False is REQUIRED — z-scoring removes the µV scale that the
# fixed ÷100 normalisation relies on.
LABRAM_PREPROCESS_CONFIG: dict = {
    "l_freq": 0.1,
    "h_freq": 75.0,
    "apply_zscore": False,
}

# CBraMod (Wang et al. 2024) — pretrained on TUAB at 200 Hz.
# Bandpass: 0.5–75 Hz (per the TUAB preprocessing in the paper).
# The patch embedding computes an explicit FFT spectral projection over 101
# frequency bins (0–100 Hz at 200 Hz sampling).  Passing 4–40 Hz filtered
# data zeros out bins 40–75 Hz, corrupting the pretrained spectral_proj
# weights — the most critical preprocessing mismatch in the codebase.
# CBraModBackbone.get_features() applies its own per-channel z-score, so
# apply_zscore can be either value; False avoids a redundant pass.
CBRAMOD_PREPROCESS_CONFIG: dict = {
    "l_freq": 0.5,
    "h_freq": 75.0,
    "apply_zscore": False,
}

# NeuroGPT (Cui et al. 2023) — pretrained on Temple University EEG Corpus.
# Bandpass: 0.5–40 Hz (per the paper's preprocessing description).
# The model applies its own per-channel z-score in get_features().
# Minor fix: adds the delta/theta band (0.5–4 Hz) missing from the default.
NEUROGPT_PREPROCESS_CONFIG: dict = {
    "l_freq": 0.5,
    "h_freq": 40.0,
    "apply_zscore": False,
}

# MIRepNet (starself/MIRepNet) — pretrained on MI datasets (BCIC-IV, OpenBMI).
# Bandpass: 4–40 Hz — the standard MI band that matches default preprocessing.
# The model applies per-channel z-score in get_features(); apply_zscore=False
# avoids a redundant pass but has no effect on model accuracy.
MIREPNET_PREPROCESS_CONFIG: dict = {
    "l_freq": 4.0,
    "h_freq": 40.0,
    "apply_zscore": False,
}

# Lookup table: backbone name → (preprocessing config, cache directory suffix).
# Backbones not listed use the default pipeline (4–40 Hz, z-scored).
BACKBONE_PREPROCESS_CONFIGS: dict[str, dict] = {
    "labram":   LABRAM_PREPROCESS_CONFIG,
    "cbramod":  CBRAMOD_PREPROCESS_CONFIG,
    "neurogpt": NEUROGPT_PREPROCESS_CONFIG,
    "mirepnet": MIREPNET_PREPROCESS_CONFIG,
}

# Cache directory suffix per backbone (appended to the base cache path).
BACKBONE_CACHE_SUFFIX: dict[str, str] = {
    "labram":   "labram",
    "cbramod":  "cbramod",
    "neurogpt": "neurogpt",
    "mirepnet": "mirepnet",
}

# Target sampling rate per backbone (Hz). Backbones not listed use 200 Hz.
#
# BCIC-IV 2a is recorded natively at 250 Hz. MIRepNet and NeuroGPT were
# pretrained at 250 Hz, so feeding them 250 Hz data avoids the lossy
# 250 -> 200 -> 250 resampling round-trip and matches the exact rate (sample
# alignment, filter phase) their checkpoints expect.
#
# LaBraM and CBraMod were pretrained at 200 Hz, so they stay at 200 Hz (their
# 0.1-75 Hz / 0.5-75 Hz content is fully preserved below the 100 Hz Nyquist).
# NOTE: CBraMod does no internal resampling, so it MUST be fed its native rate.
#
# Caches are per-backbone (BACKBONE_CACHE_SUFFIX), so the 250 Hz MIRepNet/
# NeuroGPT caches never collide with the 200 Hz LaBraM/CBraMod caches. After
# changing a backbone's rate, delete its stale cache dir so it recomputes.
BACKBONE_TARGET_SFREQ: dict[str, float] = {
    "mirepnet": 250.0,
    "neurogpt": 250.0,
    "labram":   200.0,
    "cbramod":  200.0,
}
