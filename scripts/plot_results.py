"""
Plot k-minutes vs Mean BCA for all backbones.
- Foundation backbones: loaded from results/foundation/
- Specialist backbones: loaded from results/bciciv2a/
One figure per backbone, saved to results/figures/.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOUNDATION_ROOT = ROOT / "results/foundation"
SPECIALIST_ROOT = ROOT / "results/specialist"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOUNDATION_BACKBONES = ["mirepnet", "labram", "cbramod", "neurogpt"]
SPECIALIST_BACKBONES = ["eegnet", "shallowconv", "conformer"]

# Color + linestyle keyed by *base* method name (after stripping foundation_sft_/foundation_ prefix)
METHOD_STYLES: dict[str, dict] = {
    "loso":         {"color": "#1f77b4", "ls": "--"},  # blue
    "ea":           {"color": "#ff7f0e", "ls": "--"},  # orange
    "tta":          {"color": "#2ca02c", "ls": "--"},  # green
    "finetune":     {"color": "#9467bd", "ls": "-"},   # purple
    "lora":         {"color": "#17becf", "ls": "-"},   # cyan
    "ea_lora":      {"color": "#bcbd22", "ls": "-"},   # olive
    "cld":             {"color": "#d62728", "ls": "-"},   # red
    "ea_cld":          {"color": "#8c564b", "ls": "-"},   # brown
    "anchored_cld":    {"color": "#393b79", "ls": "-"},   # dark indigo
    "ea_anchored_cld": {"color": "#7f7f7f", "ls": "-"},   # gray
    "linear_probe":    {"color": "#e377c2", "ls": "-"},   # pink
}

BACKBONE_DISPLAY = {
    "eegnet":      "EEGNet",
    "shallowconv": "ShallowConvNet",
    "conformer":   "EEG-Conformer",
    "mirepnet":    "MirepNet",
    "labram":      "LaBraM",
    "cbramod":     "CBraMod",
    "neurogpt":    "NeuroGPT",
}


def base_method(method: str) -> str:
    for prefix in ("foundation_sft_", "foundation_"):
        if method.startswith(prefix):
            return method[len(prefix):]
    return method


def load_curves(path: Path) -> dict[str, dict[float, float]]:
    """Returns {method: {k_minutes: mean_bca_across_subjects}}."""
    with open(path) as f:
        summary = json.load(f)

    # Accumulate: method -> k -> [bca, ...]
    acc: dict[str, dict[float, list[float]]] = {}
    for key, k_results in summary.items():
        method = key.split("/")[0]
        for k_str, metrics in k_results.items():
            bca = metrics.get("bca")
            if bca is None:
                continue
            acc.setdefault(method, {}).setdefault(float(k_str), []).append(bca)

    return {
        method: {k: float(np.mean(vals)) for k, vals in k_map.items()}
        for method, k_map in acc.items()
    }


def plot_backbone(backbone: str, curves: dict[str, dict[float, float]]):
    display = BACKBONE_DISPLAY.get(backbone, backbone.capitalize())
    title = f"{display} Calibration Performance Sweep ($k$-minutes vs Mean BCA)"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort: supervised (multi-k) first, then unsupervised (k=0 only)
    def sort_key(m):
        return (set(curves[m].keys()) == {0.0}, m)

    for method in sorted(curves.keys(), key=sort_key):
        km = curves[method]
        xs = sorted(km.keys())
        ys = [km[x] for x in xs]
        bm = base_method(method)
        style = METHOD_STYLES.get(bm, {"color": None, "ls": "-"})

        ax.plot(
            xs, ys,
            marker="o", markersize=5, linewidth=2,
            label=method,
            color=style["color"],
            linestyle=style["ls"],
        )

    ax.set_xlabel("Calibration Window ($k$ minutes)", fontsize=12)
    ax.set_ylabel("Mean Balanced Classification Accuracy (BCA)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend(
        title="Adaptation Method",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()
    out_path = OUT_DIR / f"{backbone}_kmin_bca.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    for backbone in FOUNDATION_BACKBONES:
        path = FOUNDATION_ROOT / backbone / "modal_summary.json"
        if not path.exists():
            print(f"Skipping {backbone}: {path} not found")
            continue
        plot_backbone(backbone, load_curves(path))

    for backbone in SPECIALIST_BACKBONES:
        path = SPECIALIST_ROOT / backbone / "modal_summary.json"
        if not path.exists():
            print(f"Skipping {backbone}: {path} not found")
            continue
        plot_backbone(backbone, load_curves(path))

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
