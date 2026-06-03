"""Plot fit_time vs K, faceted by model (one panel per backbone),
one line per adaptation. Log y-axis. K-spanning methods only."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "fit_time_by_k.csv")
df["method"] = df["method"].str.replace("foundation_sft_", "", regex=False)

ADAPTATIONS = ["finetune", "lora", "ea_lora", "cld", "ea_cld",
               "kadaptive_anchored_cld", "ea_kadaptive_anchored_cld"]
LABELS = {"finetune": "Fine-tune", "lora": "LoRA", "ea_lora": "EA+LoRA",
          "cld": "CHA", "ea_cld": "EA+CHA",
          "kadaptive_anchored_cld": "A-CHA", "ea_kadaptive_anchored_cld": "EA+A-CHA"}
df = df[df["method"].isin(ADAPTATIONS)]

k_cols = ["K=1", "K=2", "K=5", "K=10", "K=15"]
k_vals = [1, 2, 5, 10, 15]

models = sorted(df["backbone"].unique())
cmap = plt.get_cmap("tab10")
mcolor = {m: cmap(i) for i, m in enumerate(ADAPTATIONS)}

ncols = 4
nrows = -(-len(models) // ncols)  # ceil
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                         sharey=True)
axes = axes.flatten()

for ax, model in zip(axes, models):
    sub = df[df["backbone"] == model]
    for method in ADAPTATIONS:
        row = sub[sub["method"] == method]
        if row.empty:
            continue
        row = row.iloc[0]
        ys = [row[c] for c in k_cols]
        xs = [k for k, y in zip(k_vals, ys) if pd.notna(y)]
        yv = [y for y in ys if pd.notna(y)]
        if xs:
            ax.plot(xs, yv, marker="o", ms=4, lw=1.6,
                    color=mcolor[method], label=LABELS.get(method, method))
    ax.set_title(model)
    ax.set_xlabel("K (calibration minutes)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

# hide unused panels
for ax in axes[len(models):]:
    ax.set_visible(False)

for r in range(nrows):
    axes[r * ncols].set_ylabel("Mean fit_time (s)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(ADAPTATIONS),
           bbox_to_anchor=(0.5, 1.04), frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "fit_time_vs_k_by_model.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT_DIR / 'fit_time_vs_k_by_model.png'}")
