"""Plotting: calibration dose-response curves, bar charts, violin plots."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from evaluation.results import load_results
from evaluation.metrics import compute_k_star


# Color palette (colorblind-friendly)
METHOD_COLORS = {
    "loso": "#4477AA",
    "ea": "#66CCEE",
    "tta": "#228833",
    "finetune": "#CCBB44",
    "lora": "#EE6677",
    "ea_lora": "#AA3377",
}

METHOD_LABELS = {
    "loso": "Zero-shot LOSO",
    "ea": "EA (unsup.)",
    "tta": "TTA (unsup.)",
    "finetune": "Fine-tune",
    "lora": "LoRA",
    "ea_lora": "EA + LoRA",
}


def _load_method_curve(
    output_dir: str,
    dataset: str,
    backbone: str,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load BCA curve for a method: returns (k_vals, mean_bca, ci_lo, ci_hi).

    Aggregates across subjects.
    """
    results = load_results(output_dir, dataset, backbone, method)
    if not results:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Group by k_minutes
    by_k: dict[float, list] = {}
    for r in results:
        k = float(r.get("k_minutes", 0.0))
        bca = r.get("bca")
        ci_lo = r.get("ci_lo", bca)
        ci_hi = r.get("ci_hi", bca)
        if bca is not None:
            by_k.setdefault(k, []).append((bca, ci_lo, ci_hi))

    if not by_k:
        return np.array([]), np.array([]), np.array([]), np.array([])

    k_sorted = sorted(by_k.keys())
    means = [np.mean([x[0] for x in by_k[k]]) for k in k_sorted]
    lo = [np.mean([x[1] for x in by_k[k]]) for k in k_sorted]
    hi = [np.mean([x[2] for x in by_k[k]]) for k in k_sorted]

    return np.array(k_sorted), np.array(means), np.array(lo), np.array(hi)


def plot_calibration_curves(
    output_dir: str,
    dataset: str,
    backbone: str,
    methods: list[str] | None = None,
    within_subject_bca: float | None = None,
    log_x: bool = False,
    save_dir: str | None = None,
) -> None:
    """Dose-response calibration curve: BCA vs K minutes.

    - One curve per method with 95% CI shading
    - Within-subject ceiling as dashed horizontal line
    - K* annotated per curve
    """
    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if save_dir is None:
        save_dir = str(Path(output_dir) / dataset / backbone)

    fig, ax = plt.subplots(figsize=(10, 6))

    legend_handles = []

    for method in methods:
        k_vals, mean_bca, ci_lo, ci_hi = _load_method_curve(output_dir, dataset, backbone, method)
        if len(k_vals) == 0:
            continue

        color = METHOD_COLORS.get(method, "#888888")
        label = METHOD_LABELS.get(method, method)

        ax.plot(k_vals, mean_bca, color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.fill_between(k_vals, ci_lo, ci_hi, color=color, alpha=0.15)

        # Annotate K*
        if within_subject_bca is not None:
            k_star = compute_k_star(list(k_vals), list(mean_bca), within_subject_bca)
            if k_star is not None:
                ax.axvline(x=k_star, color=color, linestyle=":", linewidth=1, alpha=0.6)
                ax.text(k_star, ax.get_ylim()[0] + 0.02, f"K*={k_star:.1f}",
                        color=color, fontsize=7, rotation=90, va="bottom")

        legend_handles.append(mpatches.Patch(color=color, label=label))

    # Within-subject ceiling
    if within_subject_bca is not None:
        ax.axhline(y=within_subject_bca, color="black", linestyle="--", linewidth=1.5,
                   label="Within-subject ceiling")
        legend_handles.append(mpatches.Patch(color="black", label="Within-subject ceiling"))

    if log_x:
        ax.set_xscale("log")
        ax.set_xlabel("Calibration data (minutes, log scale)")
    else:
        ax.set_xlabel("Calibration data (minutes)")

    ax.set_ylabel("Balanced Classification Accuracy (BCA)")
    ax.set_title(f"Calibration Efficiency — {dataset} / {backbone}")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    scale_tag = "log" if log_x else "linear"
    for ext in ("pdf", "png"):
        path = Path(save_dir) / f"calibration_curve_{scale_tag}.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_bar_chart(
    output_dir: str,
    dataset: str,
    backbone: str,
    k_minutes_compare: list[float] | None = None,
    methods: list[str] | None = None,
    save_dir: str | None = None,
) -> None:
    """Bar chart comparing methods at K in {1, 2, 5} minutes."""
    if k_minutes_compare is None:
        k_minutes_compare = [1.0, 2.0, 5.0]
    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if save_dir is None:
        save_dir = str(Path(output_dir) / dataset / backbone)

    fig, axes = plt.subplots(1, len(k_minutes_compare), figsize=(4 * len(k_minutes_compare), 5),
                             sharey=True)
    if len(k_minutes_compare) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_minutes_compare):
        bcas, labels, colors, errs = [], [], [], []
        for method in methods:
            k_vals, mean_bca, ci_lo, ci_hi = _load_method_curve(output_dir, dataset, backbone, method)
            if len(k_vals) == 0:
                continue
            # Find closest K
            idx = np.argmin(np.abs(k_vals - k))
            if abs(k_vals[idx] - k) < 0.1:
                bcas.append(mean_bca[idx])
                errs.append([(mean_bca[idx] - ci_lo[idx]), (ci_hi[idx] - mean_bca[idx])])
                labels.append(METHOD_LABELS.get(method, method))
                colors.append(METHOD_COLORS.get(method, "#888888"))

        x = np.arange(len(bcas))
        err_arr = np.array(errs).T if errs else None
        bars = ax.bar(x, bcas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        if err_arr is not None:
            ax.errorbar(x, bcas, yerr=err_arr, fmt="none", color="black", capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"K = {k} min")
        ax.set_ylabel("BCA" if ax == axes[0] else "")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Method Comparison — {dataset} / {backbone}", y=1.02)
    fig.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = Path(save_dir) / f"bar_comparison.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_violin(
    output_dir: str,
    dataset: str,
    backbone: str,
    k_minutes_target: float = 5.0,
    methods: list[str] | None = None,
    save_dir: str | None = None,
) -> None:
    """Violin plot of per-subject BCA distribution at K=5 for each method."""
    if methods is None:
        methods = list(METHOD_LABELS.keys())
    if save_dir is None:
        save_dir = str(Path(output_dir) / dataset / backbone)

    fig, ax = plt.subplots(figsize=(10, 5))

    all_data = []
    all_positions = []
    all_labels = []
    all_colors = []

    for i, method in enumerate(methods):
        results = load_results(output_dir, dataset, backbone, method)
        # Collect per-subject BCA at the target K
        subject_bcas = []
        for r in results:
            k = float(r.get("k_minutes", 0.0))
            if abs(k - k_minutes_target) < 0.1:
                bca = r.get("bca")
                if bca is not None:
                    subject_bcas.append(float(bca))

        if subject_bcas:
            all_data.append(subject_bcas)
            all_positions.append(i)
            all_labels.append(METHOD_LABELS.get(method, method))
            all_colors.append(METHOD_COLORS.get(method, "#888888"))

    if all_data:
        parts = ax.violinplot(all_data, positions=all_positions, showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], all_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for part_name in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1)

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=30, ha="right")
    ax.set_ylabel("BCA (per subject)")
    ax.set_title(f"Per-Subject BCA at K={k_minutes_target} min — {dataset} / {backbone}")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = Path(save_dir) / f"violin_k{k_minutes_target:.0f}.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def generate_all_figures(
    output_dir: str,
    dataset: str,
    backbone: str,
    methods: list[str] | None = None,
    within_subject_bca: float | None = None,
) -> None:
    """Generate all standard figures for a dataset/backbone combination."""
    print(f"\nGenerating figures for {dataset}/{backbone} ...")
    plot_calibration_curves(output_dir, dataset, backbone, methods, within_subject_bca, log_x=False)
    plot_calibration_curves(output_dir, dataset, backbone, methods, within_subject_bca, log_x=True)
    plot_bar_chart(output_dir, dataset, backbone, methods=methods)
    plot_violin(output_dir, dataset, backbone, methods=methods)
    print("All figures generated.")
