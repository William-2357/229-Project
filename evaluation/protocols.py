"""Evaluation protocols: within-subject CV, LOSO, K-minute calibration sweep."""

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .metrics import compute_all_metrics


# seconds per class per trial (4-second epochs at 11 classes)
_EPOCH_SEC = 4.0


def minutes_to_trials(k_minutes: float, n_classes: int, sfreq: float = 200.0,
                       epoch_len_sec: float = 4.0) -> int:
    """Convert K minutes of calibration data to number of trials.

    Assumes balanced sampling across classes.
    Total duration = n_trials * epoch_len_sec seconds.
    """
    if k_minutes == 0.0:
        return 0
    total_sec = k_minutes * 60.0
    n_trials = int(total_sec / epoch_len_sec)
    return max(n_trials, n_classes)  # at least one trial per class


def sample_calibration_set(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_trials calibration trials, stratified by class.

    Returns (X_cal, y_cal, remaining_mask) where remaining_mask is bool
    array marking trials NOT used for calibration.
    """
    classes = np.unique(y)
    n_per_class = max(1, n_trials // len(classes))
    cal_indices = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n_take = min(n_per_class, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        cal_indices.extend(chosen.tolist())

    cal_indices = np.array(cal_indices)
    remaining_mask = np.ones(len(X), dtype=bool)
    remaining_mask[cal_indices] = False

    return X[cal_indices], y[cal_indices], remaining_mask


def within_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    adapter_class,
    adapter_kwargs: dict,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """5-fold within-subject cross-validation (ceiling estimate).

    No source data or adaptation: trains and tests on same subject's data.
    Returns aggregated metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    fold_times = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        adapter = adapter_class(seed=seed + fold_i, **adapter_kwargs)
        t0 = time.time()
        adapter.fit(source_data=(X_tr, y_tr), target_unlabeled=None, target_labeled=None)
        fold_times.append(time.time() - t0)

        preds = adapter.predict(X_te)
        all_true.extend(y_te.tolist())
        all_pred.extend(preds.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    metrics = compute_all_metrics(y_true, y_pred)
    metrics["mean_fit_time"] = float(np.mean(fold_times))
    metrics["protocol"] = "within_subject_cv"
    return metrics


def loso_evaluation(
    dataset,
    subject_id: int,
    adapter_class,
    adapter_kwargs: dict,
    seed: int = 42,
) -> dict:
    """Zero-shot LOSO: train on N-1 source subjects, evaluate on target.

    Returns metrics dict.
    """
    X_src, y_src = dataset.get_source_data(held_out_subject=subject_id)
    (_, _), (X_te, y_te) = dataset.get_target_data(subject_id)

    # Pass X_te as unlabeled target data — valid for EA/TTA which use no labels.
    # LOSOAdapter ignores this argument.
    adapter = adapter_class(seed=seed, **adapter_kwargs)
    t0 = time.time()
    adapter.fit(source_data=(X_src, y_src), target_unlabeled=X_te, target_labeled=None)
    fit_time = time.time() - t0

    preds = adapter.predict(X_te)
    metrics = compute_all_metrics(y_te, preds)
    metrics["fit_time"] = fit_time
    metrics["protocol"] = "loso"
    metrics["subject_id"] = int(subject_id)
    return metrics


def k_minute_sweep(
    dataset,
    subject_id: int,
    adapter_class,
    adapter_kwargs: dict,
    k_minutes_list: list[float],
    n_repeats: int = 5,
    seed: int = 42,
    n_classes: int | None = None,
    epoch_len_sec: float = 4.0,
    source_cache: dict | None = None,
) -> dict[float, list[dict]]:
    """K-minute calibration sweep for one subject.

    For each K in k_minutes_list, repeats n_repeats times with different random seeds.
    Source data = pooled N-1 subjects.
    Calibration data = sampled from target subject's calibration pool.
    Test data = held-out split from dataset.get_target_data().

    Returns: {k_minutes: [metrics_dict, ...]} one dict per repeat.
    """
    X_src, y_src = dataset.get_source_data(held_out_subject=subject_id)
    (X_cal_pool, y_cal_pool), (X_te, y_te) = dataset.get_target_data(subject_id)
    X_unlabeled = X_cal_pool  # all target trials as unlabeled context

    # Per-subject (subject-grouped) source for subject-aware adapters — e.g. the K-adaptive
    # anchored CLD's per-pattern prior a_i ~ 1/Var_s(v_i). Additive + cached: other adapters
    # ignore this key, and it depends only on the source split (same across K/repeats).
    if source_cache is not None and "source_per_subject" not in source_cache:
        try:
            src_subj = [s for s in dataset.subject_ids if s != subject_id]
            source_cache["source_per_subject"] = [
                dataset.get_subject_data(s, sessions=[1]) for s in src_subj
            ]
        except Exception:
            pass  # dataset without per-subject access -> adaptive mode falls back to isotropic

    if n_classes is None:
        n_classes = len(np.unique(y_src))

    results: dict[float, list[dict]] = {}

    for k in k_minutes_list:
        n_cal_trials = minutes_to_trials(k, n_classes, epoch_len_sec=epoch_len_sec)
        repeats = []

        for repeat in range(n_repeats):
            rng = np.random.default_rng(seed + repeat * 1000)
            if n_cal_trials > 0:
                X_cal, y_cal, _ = sample_calibration_set(
                    X_cal_pool, y_cal_pool, n_cal_trials, rng
                )
            else:
                X_cal, y_cal = np.empty((0, *X_cal_pool.shape[1:]), dtype=X_cal_pool.dtype), np.empty(0, dtype=y_cal_pool.dtype)

            adapter = adapter_class(seed=seed, **adapter_kwargs)
            t0 = time.time()
            adapter.fit(
                source_data=(X_src, y_src),
                target_unlabeled=X_unlabeled,
                target_labeled=(X_cal, y_cal) if n_cal_trials > 0 else None,
                source_cache=source_cache,
            )
            fit_time = time.time() - t0
            # Pure on-target training time (ADMM solve / finetune epoch loop only),
            # excluding backbone load, cache reads, and feature extraction. 0.0 for
            # adapters that don't adapt on target data. See BaseAdapter.train_time.
            train_fit_time = float(getattr(adapter, "train_time", 0.0) or 0.0)

            preds = adapter.predict(X_te)
            m = compute_all_metrics(y_te, preds)
            m["fit_time"] = fit_time
            m["train_fit_time"] = train_fit_time
            m["k_minutes"] = float(k)
            m["n_cal_trials"] = int(len(X_cal))
            m["repeat"] = repeat
            m["protocol"] = "k_minute_sweep"
            m["subject_id"] = int(subject_id)
            repeats.append(m)

        results[k] = repeats

    return results


def aggregate_across_subjects(
    per_subject_results: list[dict],
) -> dict:
    """Average BCA and CI across subjects.

    Args:
        per_subject_results: list of metrics dicts (one per subject)

    Returns:
        dict with mean_bca, std_bca, mean_ci_lo, mean_ci_hi, n_subjects
    """
    bcas = [r["bca"] for r in per_subject_results]
    return {
        "mean_bca": float(np.mean(bcas)),
        "std_bca": float(np.std(bcas)),
        "mean_ci_lo": float(np.mean([r["ci_lo"] for r in per_subject_results])),
        "mean_ci_hi": float(np.mean([r["ci_hi"] for r in per_subject_results])),
        "n_subjects": len(per_subject_results),
        "per_subject": per_subject_results,
    }
