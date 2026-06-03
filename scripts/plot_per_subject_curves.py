"""Per-subject BCA vs K, faceted by adaptation method, for one backbone (default neurogpt).

One panel per method; x = calibration minutes (K), y = BCA, one line per subject.
Methods: Fine-tune, LoRA, CHA, A-CHA. No figure title (paper figure).
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"; OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, method_key, results_dir)
METHODS = [
    ("Fine-tune", "foundation_sft_finetune",                "results/foundation"),
    ("LoRA",      "foundation_sft_lora",                    "results/foundation"),
    ("CHA",       "foundation_sft_cld",                     "results/foundation"),
    ("A-CHA",     "foundation_sft_kadaptive_anchored_cld",  "results/kadaptive_fixed"),
]
EXCLUDE_K = {30.0}


def per_subject(results_dir, backbone, method, metric="bca"):
    d = json.load(open(f"{results_dir}/{backbone}/modal_summary.json"))
    out = {}  # sid -> {K: val}
    for key, cells in d.items():
        if key.split("/")[0] != method:
            continue
        sid = int(key.split("/")[1].split("_")[1])
        out[sid] = {float(k): r[metric] for k, r in cells.items()
                    if isinstance(r, dict) and metric in r
                    and float(k) > 0 and float(k) not in EXCLUDE_K}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="neurogpt")
    ap.add_argument("--metric", default="bca")
    args = ap.parse_args()

    subs = list(range(1, 10))
    cmap = plt.get_cmap("tab10")
    scolor = {s: cmap((s - 1) % 10) for s in subs}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (label, mkey, rdir) in zip(axes, METHODS):
        data = per_subject(rdir, args.backbone, mkey, args.metric)
        for s in subs:
            if s not in data:
                continue
            ks = sorted(data[s])
            ax.plot(ks, [data[s][k] for k in ks], marker="o", ms=4, lw=1.4,
                    color=scolor[s], label=f"S{s}")
        ax.axhline(0.25, color="gray", ls=":", lw=0.9)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for ax in axes[2:]:
        ax.set_xlabel("Minutes of Target Data Available", fontsize=11, fontweight="bold")
    ylab = "Balanced Class Accuracy (BCA)" if args.metric == "bca" else "Cohen's $\\kappa$"
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(ylab, fontsize=11, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Subject",
               fontsize=10, bbox_to_anchor=(1.06, 0.5))
    plt.tight_layout()
    out = OUT_DIR / f"{args.backbone}_per_subject_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
