"""Evaluation metrics: BCA, Cohen's kappa, block-bootstrap CI, permutation null, K*."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred))


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    block_size: int = 30,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Block-bootstrap 95% CI for BCA.

    Blocks of `block_size` trials are resampled to preserve temporal autocorrelation.

    Returns:
        (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    # ceil ensures the bootstrap sample covers ~n trials regardless of block_size.
    # floor division with small n (e.g. 58 trials, block_size=30) gives n_blocks=1,
    # making every replicate identical and the CI degenerate (lo == hi).
    n_blocks = max(1, int(np.ceil(n / block_size)))
    block_starts = np.arange(0, n_blocks * block_size, block_size)

    boot_stats = []
    for _ in range(n_resamples):
        chosen = rng.choice(block_starts, size=n_blocks, replace=True)
        indices = np.concatenate([np.arange(s, min(s + block_size, n)) for s in chosen])
        indices = indices[:n]
        bca = balanced_accuracy_score(y_true[indices], y_pred[indices])
        boot_stats.append(bca)

    alpha = 1 - confidence
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def permutation_null(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[float, bool]:
    """Shuffled-label permutation test for BCA.

    Returns:
        (p_value, is_significant) where significance threshold is 0.05
    """
    rng = np.random.default_rng(seed)
    observed = balanced_accuracy_score(y_true, y_pred)
    null_dist = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_true)
        null_dist.append(balanced_accuracy_score(shuffled, y_pred))
    p_value = float(np.mean(np.array(null_dist) >= observed))
    return p_value, p_value < 0.05


def compute_k_star(
    k_values: list[float],
    bca_values: list[float],
    ceiling: float,
    threshold_fraction: float = 0.80,
) -> float | None:
    """Compute K* = minimum K where BCA >= threshold_fraction * ceiling.

    Returns None if threshold is never reached.
    """
    threshold = threshold_fraction * ceiling
    for k, bca in zip(k_values, bca_values):
        if bca >= threshold:
            return float(k)
    return None


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    n_permutations: int = 200,
    block_size: int = 30,
    seed: int = 42,
) -> dict:
    """Compute full metric suite for one condition."""
    bca = balanced_accuracy(y_true, y_pred)
    kappa = cohens_kappa(y_true, y_pred)
    ci_lo, ci_hi = block_bootstrap_ci(y_true, y_pred, n_resamples=n_bootstrap,
                                       block_size=block_size, seed=seed)
    p_val, significant = permutation_null(y_true, y_pred, n_permutations=n_permutations, seed=seed)
    return {
        "bca": bca,
        "kappa": kappa,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_val,
        "significant": significant,
        "n_trials": int(len(y_true)),
    }
