"""k-minutes BCA plots for the specialist backbones (eegnet/shallowconv/conformer) from the
existing per-subject results: convex (cld, ea_cld) vs lora, ea_lora, finetune."""
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
BACKBONES = ["eegnet", "shallowconv", "conformer"]
KS = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]
METHODS = [  # (dir, label, color, marker)
    ("cld",      "convex (CLD)",   "tab:red",    "o"),
    ("ea_cld",   "EA + convex",    "tab:green",  "D"),
    ("lora",     "LoRA",           "tab:orange", "s"),
    ("ea_lora",  "EA-LoRA",        "tab:purple", "v"),
    ("finetune", "finetune",       "tab:blue",   "^"),
]


def curve(backbone, method):
    ys = []
    for k in KS:
        vals = []
        for s in range(1, 10):
            f = REPO / f"results/bciciv2a/{backbone}/{method}/subject_{s:02d}_k{k}.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["bca"])
        ys.append(np.mean(vals) if vals else np.nan)
    return ys


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    summary = {}
    for ax, b in zip(axes, BACKBONES):
        summary[b] = {}
        for m, lab, c, mk in METHODS:
            ys = curve(b, m)
            summary[b][m] = ys
            lw = 2.4 if m in ("cld", "ea_cld") else 1.7
            ax.plot(KS, ys, marker=mk, color=c, lw=lw, label=f"{lab} ({np.nanmean(ys):.3f})")
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("Calibration window K (minutes)")
        ax.set_title(f"{b}", fontsize=12)
        ax.grid(True, alpha=0.3, which="both"); ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Mean test BCA (9 subjects)")
    fig.suptitle("Specialist backbones — calibration-efficiency sweep "
                 "(convex CLD / EA+convex vs LoRA / EA-LoRA / finetune)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPO / "research" / "specialists_kmin.png"
    fig.savefig(out, dpi=140); print(f"saved {out}")

    # also per-backbone standalone + a text summary table
    for b in BACKBONES:
        f2, a = plt.subplots(figsize=(7, 5))
        for m, lab, c, mk in METHODS:
            ys = summary[b][m]
            a.plot(KS, ys, marker=mk, color=c, lw=2.4 if m in ("cld", "ea_cld") else 1.7,
                   label=f"{lab} ({np.nanmean(ys):.3f})")
        a.set_xscale("log"); a.set_xticks(KS); a.set_xticklabels([str(k) for k in KS])
        a.set_xlabel("Calibration window K (minutes)"); a.set_ylabel("Mean test BCA (9 subjects)")
        a.set_title(f"{b} — calibration sweep"); a.grid(True, alpha=0.3, which="both")
        a.legend(loc="lower right", fontsize=9); f2.tight_layout()
        f2.savefig(REPO / f"research/specialist_{b}_kmin.png", dpi=140)
    print("\n=== specialist mean BCA over K (per method) ===")
    print(f"{'backbone':>11} | " + " | ".join(f"{m:>9}" for m,_,_,_ in METHODS))
    for b in BACKBONES:
        print(f"{b:>11} | " + " | ".join(f"{np.nanmean(summary[b][m]):>9.3f}" for m,_,_,_ in METHODS))


if __name__ == "__main__":
    main()
