"""Per-subject BCA heatmap across adaptation methods for one backbone (default neurogpt).

Rows = methods, columns = subjects (+ Mean); cell = mean BCA over K>0 per subject.
Red box marks the best method per subject. No title / footnote (paper figure).

Usage: python scripts/plot_per_subject.py [--backbone neurogpt] [--metric bca]
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"; OUT_DIR.mkdir(parents=True, exist_ok=True)
FOUNDATION = {"neurogpt", "mirepnet", "labram"}

ROWS = [("Fine-tune", "finetune"), ("LoRA", "lora"), ("EA+LoRA", "ea_lora"),
        ("CHA", "cld"), ("EA+CHA", "ea_cld"),
        ("K-Anchored-CHA", "kadaptive_anchored_cld"),
        ("EA+K-Anchored-CHA", "ea_kadaptive_anchored_cld")]
KADAPT = {"kadaptive_anchored_cld", "ea_kadaptive_anchored_cld"}


def base(m):
    for p in ("foundation_sft_", "foundation_"):
        if m.startswith(p):
            return m[len(p):]
    return m


def load(path, metric):
    d = json.load(open(path)); out = {}
    for key, cells in d.items():
        bm = base(key.split("/")[0]); sid = int(key.split("/")[1].split("_")[1])
        vals = [r[metric] for k, r in cells.items()
                if isinstance(r, dict) and metric in r and float(k) > 0]
        if vals:
            out.setdefault(bm, {})[sid] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="neurogpt")
    ap.add_argument("--metric", default="bca")
    args = ap.parse_args()
    bb = args.backbone
    old_root = "results/foundation" if bb in FOUNDATION else "results/specialist"
    old = load(f"{old_root}/{bb}/modal_summary.json", args.metric)
    kad = load(f"results/kadaptive_fixed/{bb}/modal_summary.json", args.metric)

    subs = list(range(1, 10))
    M = np.array([[(kad if m in KADAPT else old).get(m, {}).get(s, np.nan) for s in subs]
                  for _, m in ROWS])
    G = np.hstack([M, np.nanmean(M, axis=1, keepdims=True)])
    labels = [r[0] for r in ROWS]
    cols = [f"S{s}" for s in subs] + ["Mean"]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    im = ax.imshow(G, cmap="viridis", aspect="auto",
                   vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(len(subs) - 0.5, color="white", lw=3)
    for i in range(len(labels)):
        for j in range(len(cols)):
            v = G[i, j]
            if v != v:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v < 0.5 else "black")
    for j in range(len(subs)):
        col = M[:, j]
        if np.all(np.isnan(col)):
            continue
        bi = int(np.nanargmax(col))
        ax.add_patch(Rectangle((j - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="red", lw=2.4))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("BCA (mean over $K>0$)" if args.metric == "bca"
                   else "$\\kappa$ (mean over $K>0$)", fontsize=10)
    plt.tight_layout()
    suffix = "" if args.metric == "bca" else f"_{args.metric}"
    out = OUT_DIR / f"{bb}_per_subject_heatmap{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
