"""Before/after k-minutes figure: original modal results (convex loses) vs corrected
same-backbone results (LoRA+convex wins). Left = teammate's modal_summary; right = our runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "research" / "runs"
KS = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]


def modal_curve(method):
    d = json.loads((REPO / "results/bciciv2a/mirepnet/modal_summary.json").read_text())
    ys = []
    for k in KS:
        vals = []
        for s in range(1, 10):
            key = f"{method}/subject_{s:02d}"
            cell = d.get(key, {}).get(str(k))
            if isinstance(cell, dict) and "bca" in cell:
                vals.append(cell["bca"])
        ys.append(np.mean(vals) if vals else np.nan)
    return ys


def run_curve(tag):
    d = json.loads((RUNS / f"{tag}__full.json").read_text())["per_k"]
    return [d[next(kk for kk in d if float(kk) == k)] for k in KS]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- BEFORE (original modal results; weaker 200Hz setup) ----
before = [
    ("foundation_sft_lora",    "LoRA",            "tab:orange", "s"),
    ("foundation_sft_ea_lora", "EA-LoRA",         "tab:purple", "v"),
    ("foundation_sft_finetune","finetune",        "tab:blue",   "^"),
    ("foundation_sft_cld",     "convex (CLD)",    "tab:red",    "o"),
]
for m, lab, c, mk in before:
    ys = modal_curve(m)
    axL.plot(KS, ys, marker=mk, color=c, lw=2.0 if "cld" in m else 1.6,
             label=f"{lab} ({np.nanmean(ys):.3f})")
axL.set_title("BEFORE — original modal results\n(convex CLD loses to LoRA; dips at low K)", fontsize=11)

# ---- AFTER (corrected 250Hz, identical source-FT backbone) ----
after = [
    ("iter8_lora_convex_full", "LoRA + convex (ours)",        "tab:green",  "D", 2.6),
    ("full_lora_250",          "LoRA",                        "tab:orange", "s", 1.8),
    ("iter3_convex_full",      "convex (frozen backbone)",    "tab:red",    "o", 1.8),
    ("full_ft_250",            "finetune",                    "tab:blue",   "^", 1.8),
]
for tag, lab, c, mk, lw in after:
    ys = run_curve(tag)
    axR.plot(KS, ys, marker=mk, color=c, lw=lw, label=f"{lab} ({np.mean(ys):.3f})")
axR.set_title("AFTER — corrected 250Hz, same source-FT backbone\n(LoRA+convex wins overall)", fontsize=11)

for ax in (axL, axR):
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Calibration window K (minutes)"); ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
axL.set_ylabel("Mean test BCA (9 subjects)")
fig.suptitle("Convex head for EEG calibration on MIRepNet: from losing to LoRA → beating it\n"
             "(absolute BCA differs across panels — left uses the original weaker/200Hz setup; "
             "the story is the convex-vs-LoRA flip)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = REPO / "research" / "before_after_kmin.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
