"""Plot foundation-model modal_summary.json files in a multi-panel comparison figure.

Example:
    python scripts/plot_foundation_modal_summary.py \
      --json results/bciciv2a/labram/modal_summary.json \
             results/bciciv2a/mirepnet/modal_summary_mirepnet.json \
             results/bciciv2a/neurogpt/modal_summary_neuro.json \
      --output results/bciciv2a/foundation_backbones.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "foundation_cld": {
        "label": "CLD",
        "color": "#8e63c7",
        "marker": "D",
    },
    "foundation_ea": {
        "label": "Base EA",
        "color": "#1f77b4",
        "marker": "o",
    },
    "foundation_ea_cld": {
        "label": "EA + CLD",
        "color": "#8c564b",
        "marker": "X",
    },
    "foundation_ea_lora": {
        "label": "EA + LoRA",
        "color": "#e31a1c",
        "marker": "s",
    },
    "foundation_finetune": {
        "label": "Finetune",
        "color": "#ff7f0e",
        "marker": "v",
    },
    "foundation_lora": {
        "label": "LoRA",
        "color": "#1ca02c",
        "marker": "^",
    },
}

METHOD_ORDER = [
    "foundation_cld",
    "foundation_ea",
    "foundation_ea_cld",
    "foundation_ea_lora",
    "foundation_finetune",
    "foundation_lora",
]

BACKBONE_LABELS = {
    "mirepnet": "Mirepnet",
    "neurogpt": "NeuroGPT",
    "labram": "LaBraM",
    "reve": "REVE",
}


def infer_backbone_name(json_path: Path) -> str:
    parent_name = json_path.parent.name.lower()
    if parent_name in BACKBONE_LABELS:
        return parent_name

    stem = json_path.stem.lower()
    for backbone in BACKBONE_LABELS:
        if backbone in stem:
            return backbone
    return parent_name


def load_mean_curves(json_path: Path) -> dict[str, list[tuple[float, float]]]:
    data = json.loads(json_path.read_text())
    by_method_by_k: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for _, k_results in data.items():
        for k_str, entry in k_results.items():
            method = entry.get("_meta", {}).get("method")
            if method is None:
                continue
            try:
                k_val = float(k_str)
            except ValueError:
                k_val = float(entry.get("k_minutes", 0.0))
            by_method_by_k[method][k_val].append(float(entry["bca"]))

    mean_curves: dict[str, list[tuple[float, float]]] = {}
    for method, by_k in by_method_by_k.items():
        curve = []
        for k_val in sorted(by_k):
            curve.append((k_val, float(np.mean(by_k[k_val]))))
        mean_curves[method] = curve
    return mean_curves


def plot_backbone_panel(ax: plt.Axes, backbone: str, curves: dict[str, list[tuple[float, float]]]) -> None:
    for method in METHOD_ORDER:
        if method not in curves:
            continue

        style = METHOD_STYLES[method]
        curve = curves[method]
        x_vals = [x for x, _ in curve]
        y_vals = [y for _, y in curve]

        if method == "foundation_ea":
            ax.plot(
                x_vals,
                y_vals,
                linestyle="None",
                marker=style["marker"],
                color=style["color"],
                markersize=6,
                label=style["label"],
            )
            continue

        ax.plot(
            x_vals,
            y_vals,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.8,
            markersize=6,
            label=style["label"],
        )

    ax.set_title(BACKBONE_LABELS.get(backbone, backbone), fontsize=17)
    ax.set_xlabel(r"Adaptation Window ($k_{mins}$)", fontsize=14)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one or more foundation-model modal_summary.json files."
    )
    parser.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="One or more modal_summary.json paths.",
    )
    parser.add_argument(
        "--output",
        default="results/figures/foundation_backbones_comparison.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Adaptation Methods Performance Across Different Backbones",
        help="Figure title.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    json_paths = [Path(path) for path in args.json]

    for path in json_paths:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

    backbone_specs = []
    for path in json_paths:
        backbone = infer_backbone_name(path)
        curves = load_mean_curves(path)
        backbone_specs.append((backbone, curves))

    n_panels = len(backbone_specs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 5.3), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (backbone, curves) in zip(axes, backbone_specs):
        plot_backbone_panel(ax, backbone, curves)

    axes[0].set_ylabel("Mean Balanced Class Accuracy (BCA)", fontsize=14)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.985, 0.12), frameon=True)
    fig.suptitle(args.title, fontsize=20, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
