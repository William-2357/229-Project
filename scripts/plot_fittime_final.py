"""Final fit-time plots: no-pad (solid) vs pad-256 (dashed), compile-excluded, K<=15.

Reads results/fittime/<bb>/modal_summary.json (padding=false) and
results/fittime256/<bb>/modal_summary.json (padding=256), uses fit_time_warm.
Produces:
  results/figures/fit_time_vs_k_by_model.png  - one panel per backbone
  results/figures/fit_time_vs_k.png           - averaged across models (neurogpt excluded as outlier)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BB_ORDER = [("NeuroGPT","neurogpt"),("MIRepNet","mirepnet"),("LaBraM","labram"),
            ("CBraMod","cbramod"),("EEGNet","eegnet"),("ShallowConvNet","shallowconv"),
            ("EEG-Conformer","conformer")]
GRAD = [("Fine-tune",("foundation_sft_finetune","finetune")),
        ("LoRA",("foundation_sft_lora","lora")),
        ("EA+LoRA",("foundation_sft_ea_lora","ea_lora"))]
CONV = [("CHA",("foundation_sft_cld","cld")),
        ("EA+CHA",("foundation_sft_ea_cld","ea_cld")),
        ("A-CHA",("foundation_sft_kadaptive_anchored_cld","kadaptive_anchored_cld")),
        ("EA+A-CHA",("foundation_sft_ea_kadaptive_anchored_cld","ea_kadaptive_anchored_cld"))]
ALL = GRAD + CONV
KS = [0.5, 1, 2, 5, 10, 15]
cmap = plt.get_cmap("tab10")
COL = {lab: cmap(i) for i, (lab, _) in enumerate(ALL)}


def load(d):
    return json.load(open(d))


def curve(summary, keys):
    perk = {}
    for key, cells in summary.items():
        if key.split("/")[0] not in keys:
            continue
        for k, r in cells.items():
            if isinstance(r, dict) and float(k) in KS:
                perk.setdefault(float(k), []).append(r.get("fit_time_warm", r.get("fit_time")))
    xs = [k for k in KS if k in perk]
    return xs, [np.mean(perk[k]) for k in xs]


def panel(ax, nopad, pad):
    for lab, keys in GRAD:
        xs, ys = curve(nopad, set(keys))
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=COL[lab])
    for lab, keys in CONV:
        xs, ys = curve(nopad, set(keys))
        if xs:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=COL[lab])
        xs2, ys2 = curve(pad, set(keys))
        if xs2:
            ax.plot(xs2, ys2, marker="s", ms=3, lw=1.3, ls="--", color=COL[lab])


def legend_handles():
    h = [Line2D([0], [0], color=COL[lab], lw=2, label=lab) for lab, _ in ALL]
    h += [Line2D([0], [0], color="0.3", lw=2, ls="-", label="no pad"),
          Line2D([0], [0], color="0.3", lw=2, ls="--", label="pad 256")]
    return h


# ---- faceted by model ----
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
axes = axes.flatten()
for ax, (disp, bb) in zip(axes, BB_ORDER):
    panel(ax, load(ROOT / f"results/fittime/{bb}/modal_summary.json"),
          load(ROOT / f"results/fittime256/{bb}/modal_summary.json"))
    ax.set_title(disp, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
axes[-1].set_visible(False)
for ax in axes[4:7]:
    ax.set_xlabel("Minutes of Target Data", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Minutes of Target Data", fontsize=11, fontweight="bold")  # bottom of col0 hidden? keep
for i in (0, 4):
    axes[i].set_ylabel("Fit time (s)", fontsize=11, fontweight="bold")
fig.legend(handles=legend_handles(), loc="center", bbox_to_anchor=(0.88, 0.25),
           ncol=2, fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig(OUT_DIR / "fit_time_vs_k_by_model.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved {OUT_DIR / 'fit_time_vs_k_by_model.png'}")

# ---- averaged across models (exclude neurogpt outlier) ----
avg_bb = [bb for _, bb in BB_ORDER if bb != "neurogpt"]
nopads = [load(ROOT / f"results/fittime/{bb}/modal_summary.json") for bb in avg_bb]
pads = [load(ROOT / f"results/fittime256/{bb}/modal_summary.json") for bb in avg_bb]
fig, ax = plt.subplots(figsize=(8.5, 6))


def mean_curve(summaries, keys):
    acc = {k: [] for k in KS}
    for s in summaries:
        xs, ys = curve(s, set(keys))
        for k, y in zip(xs, ys):
            acc[k].append(y)
    xs = [k for k in KS if acc[k]]
    return xs, [np.mean(acc[k]) for k in xs]


for lab, keys in GRAD:
    xs, ys = mean_curve(nopads, keys)
    if xs:
        ax.plot(xs, ys, marker="o", ms=5, lw=1.6, color=COL[lab])
for lab, keys in CONV:
    xs, ys = mean_curve(nopads, keys)
    if xs:
        ax.plot(xs, ys, marker="o", ms=5, lw=1.6, color=COL[lab])
    xs2, ys2 = mean_curve(pads, keys)
    if xs2:
        ax.plot(xs2, ys2, marker="s", ms=4, lw=1.4, ls="--", color=COL[lab])
ax.set_xlabel("Minutes of Target Data Available", fontsize=12, fontweight="bold")
ax.set_ylabel("Fit time (s, compile-excluded)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
ax.legend(handles=legend_handles(), ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "fit_time_vs_k.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved {OUT_DIR / 'fit_time_vs_k.png'}")
