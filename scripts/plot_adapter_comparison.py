"""Recreate the `adapter_comparison.png` style for the latest runs.

Reads the same modal_summary.json files that feed results/figures/ (results/foundation/
and results/specialist/) and renders, per backbone:
  - unsupervised methods (K=0 only) as dashed horizontal baselines
  - supervised methods (K>0) as learning curves with a mean +/- std band
    (std across subjects, matching unpack_and_plot.py)
  - a 4-class chance line at 0.25

Output: results/figures/{backbone}_adapter_comparison.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOUNDATION_ROOT = ROOT / "results/foundation"
SPECIALIST_ROOT = ROOT / "results/specialist"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOUNDATION_BACKBONES = ["mirepnet", "labram", "cbramod", "neurogpt"]
SPECIALIST_BACKBONES = ["eegnet", "shallowconv", "conformer"]

BACKBONE_LABELS = {
    "eegnet": "EEGNet", "shallowconv": "ShallowConvNet", "conformer": "EEG-Conformer",
    "mirepnet": "MirepNet", "labram": "LaBraM", "cbramod": "CBraMod", "neurogpt": "NeuroGPT",
}
DATASET_LABELS = {"bciciv2a": "BCI Competition IV 2a", "synthetic": "Synthetic"}

# Per-metric plot settings: y-axis label, chance level + label, filename suffix.
METRIC_CONFIG = {
    "bca": {
        "ylabel": "Balanced Class Accuracy (BCA)",
        "chance": 0.25, "chance_label": "Chance (4-class)", "suffix": "",
    },
    "kappa": {
        "ylabel": "Cohen's Kappa (κ)",
        "chance": 0.0, "chance_label": "Chance (κ=0)", "suffix": "_kappa",
    },
}

# Encoding uses TWO channels:
#   COLOR  = method family; EA variant uses a LIGHTER tint of the family color
#   SHAPE  = EA vs non-EA  (non-EA: circle;  EA: square)
# Supervised curves are always solid lines; dashed is reserved for the K=0
# baselines so EA curves don't get confused with them.


def lighten(hex_color: str, amount: float = 0.5) -> str:
    """Blend a hex color toward white by `amount` (0=same, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"

# Unsupervised (K=0 only) -> dashed horizontal baseline
COLORS_K0 = {"loso": "#1f77b4", "ea": "#ff7f0e", "tta": "#17becf"}

# Supervised (K>0): one color per *base* family; EA variant reuses the family color.
COLORS_ADAPT = {
    "finetune": "#d62728",                                # red
    "lora": "#2ca02c",       "ea_lora": "#2ca02c",        # green  (LoRA family)
    "cld": "#9467bd",        "ea_cld": "#9467bd",         # purple (CHA family)
    "anchored_cld": "#e377c2", "ea_anchored_cld": "#e377c2",  # pink (Anchored-CHA family)
    "kadaptive_anchored_cld": "#bcbd22", "ea_kadaptive_anchored_cld": "#bcbd22",  # olive (K-adaptive)
    "linear_probe": "#8c564b",                            # brown
}

# Base method names (post-prefix-strip) that apply Euclidean Alignment.
EA_METHODS = {"ea", "ea_lora", "ea_cld", "ea_anchored_cld", "ea_kadaptive_anchored_cld"}
LABELS = {
    "loso": "LOSO", "ea": "EA", "tta": "TTA",
    "finetune": "Fine-tune", "lora": "LoRA", "ea_lora": "EA+LoRA",
    "cld": "CHA", "ea_cld": "EA+CHA",
    "anchored_cld": "Anchored-CHA", "ea_anchored_cld": "EA+Anchored-CHA",
    "kadaptive_anchored_cld": "A-CHA", "ea_kadaptive_anchored_cld": "EA+A-CHA",
    "linear_probe": "Linear Probe",
}
SUPERVISED_ORDER = ["finetune", "lora", "ea_lora", "cld", "ea_cld",
                    "kadaptive_anchored_cld", "ea_kadaptive_anchored_cld", "linear_probe"]

# K values (minutes) to drop from the supervised curves
EXCLUDE_K = {30.0}


def base_method(method: str) -> str:
    for prefix in ("foundation_sft_", "foundation_"):
        if method.startswith(prefix):
            return method[len(prefix):]
    return method


def aggregate(path: Path, metric: str = "bca"):
    """Return (curves, n_subjects).

    curves[base_method] = {k: (mean, std_across_subjects)} for the chosen metric.
    """
    summary = json.loads(path.read_text())
    # base_method -> k -> [metric per subject]
    acc: dict[str, dict[float, list[float]]] = {}
    subjects = set()
    for key, k_results in summary.items():
        bm = base_method(key.split("/")[0])
        for k_str, rec in k_results.items():
            val = rec.get(metric)
            if val is None:
                continue
            acc.setdefault(bm, {}).setdefault(float(k_str), []).append(float(val))
            if rec.get("subject_id") is not None:
                subjects.add(rec["subject_id"])
    curves = {
        bm: {k: (float(np.mean(v)), float(np.std(v))) for k, v in kmap.items()}
        for bm, kmap in acc.items()
    }
    return curves, len(subjects)


def plot_backbone(backbone: str, dataset: str, curves: dict, n_subjects: int,
                  show_bands: bool = True, metric: str = "bca", note: str = "") -> None:
    cfg = METRIC_CONFIG[metric]
    fig, ax = plt.subplots(figsize=(9, 6))

    # Unsupervised K=0 baselines as horizontal dashed lines
    for bm, color in COLORS_K0.items():
        if bm in curves and 0.0 in curves[bm]:
            mean, _ = curves[bm][0.0]
            ax.axhline(mean, color=color, linestyle="--", linewidth=1.5,
                       label=LABELS[bm], alpha=0.85)

    # Supervised learning curves (K>0) with mean +/- std band
    for bm in SUPERVISED_ORDER:
        if bm not in curves:
            continue
        ks = sorted(k for k in curves[bm] if k > 0.0 and k not in EXCLUDE_K)
        if not ks:
            continue
        means = np.array([curves[bm][k][0] for k in ks])
        stds = np.array([curves[bm][k][1] for k in ks])
        is_ea = bm in EA_METHODS
        color = COLORS_ADAPT.get(bm, "#888888")
        if is_ea:
            color = lighten(color, 0.5)
        ax.plot(ks, means,
                marker="s" if is_ea else "o",
                linestyle="-",
                markersize=5.5, linewidth=1.4,
                label=LABELS.get(bm, bm), color=color)
        if show_bands:
            ax.fill_between(ks, means - stds, means + stds,
                            alpha=0.15, color=COLORS_ADAPT.get(bm, "#888888"))

    ax.axhline(cfg["chance"], color="gray", linestyle=":", linewidth=1.0,
               label=cfg["chance_label"])
    ax.set_xlabel("Minutes of Target Data Available", fontsize=12, fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_png = OUT_DIR / f"{backbone}_adapter_comparison{cfg['suffix']}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bands", action="store_true",
                        help="Hide the mean +/- std shaded error bands.")
    parser.add_argument("--metric", choices=list(METRIC_CONFIG), default="bca",
                        help="Metric to plot (bca or kappa).")
    parser.add_argument("--kadapt-root", default=None,
                        help="Dir with {backbone}/modal_summary.json holding the K-adaptive "
                             "results on corrected (cue-aligned) data; overlaid onto the "
                             "original-data curves with an annotation.")
    args = parser.parse_args()

    kadapt_root = Path(args.kadapt_root) if args.kadapt_root else None
    KADAPT_BASES = {"kadaptive_anchored_cld", "ea_kadaptive_anchored_cld"}

    jobs = [(bb, FOUNDATION_ROOT / bb / "modal_summary.json") for bb in FOUNDATION_BACKBONES]
    jobs += [(bb, SPECIALIST_ROOT / bb / "modal_summary.json") for bb in SPECIALIST_BACKBONES]
    for backbone, path in jobs:
        if not path.exists():
            print(f"Skipping {backbone}: {path} not found")
            continue
        dataset = path.parent.parent.name if path.parent.parent.name in DATASET_LABELS else "bciciv2a"
        curves, n_subjects = aggregate(path, metric=args.metric)
        note = ""
        if kadapt_root is not None:
            kpath = kadapt_root / backbone / "modal_summary.json"
            if kpath.exists():
                kcurves, _ = aggregate(kpath, metric=args.metric)
                for bm in KADAPT_BASES:
                    if bm in kcurves:
                        curves[bm] = kcurves[bm]
                note = ("K-Anchored-CHA: corrected cue-aligned data  |  "
                        "all other methods: original (+2s-shifted) data")
            else:
                print(f"  ({backbone}: no K-adaptive overlay at {kpath})")
        plot_backbone(backbone, dataset, curves, n_subjects,
                      show_bands=not args.no_bands, metric=args.metric, note=note)
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
