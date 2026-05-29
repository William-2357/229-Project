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
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline returning (X, y).

    Steps:
        1. Common-average reference
        2. Bandpass 4-40 Hz (zero-phase FIR)
        3. Notch 50/60 Hz
        4. Resample to target_sfreq
        5. Epoch (4 s aligned to cue)
        6. Per-channel, per-trial z-score

    Returns:
        X: (N, C, T) float32
        y: (N,) int32
    """
    data = common_average_reference(raw)
    data = bandpass_filter(data, sfreq, l_freq=4.0, h_freq=40.0)
    data = notch_filter(data, sfreq, freqs=notch_freqs)
    data = resample(data, sfreq, target_sfreq)

    # Rescale event indices to new sampling rate
    scale = target_sfreq / sfreq
    scaled_events = (events * scale).astype(int)

    X = extract_epochs(data, scaled_events, target_sfreq, tmin=tmin, epoch_len_sec=epoch_len_sec)

    # Keep only valid epochs (events that fit in recording)
    n_valid = len(X)
    y = labels[:n_valid].astype(np.int32)

    X = zscore_per_trial(X)
    return X, y
