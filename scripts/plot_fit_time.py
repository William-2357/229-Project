"""Plot fit_time vs K, one line per adaptation = mean across all models.
Single panel, log y-axis. Drops neurogpt and K=0-only methods."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "fit_time_by_k.csv")

# Normalize method names: strip the foundation prefix so e.g.
# "foundation_sft_lora" overlays with conventional "lora".
df["method"] = df["method"].str.replace("foundation_sft_", "", regex=False)

# Drop neurogpt (outlier) and K=0-only methods.
df = df[df["backbone"] != "neurogpt"]
ADAPTATIONS = ["finetune", "lora", "ea_lora", "cld", "ea_cld",
               "kadaptive_anchored_cld", "ea_kadaptive_anchored_cld"]
LABELS = {"finetune": "Fine-tune", "lora": "LoRA", "ea_lora": "EA+LoRA",
          "cld": "CHA", "ea_cld": "EA+CHA",
          "kadaptive_anchored_cld": "A-CHA", "ea_kadaptive_anchored_cld": "EA+A-CHA"}
df = df[df["method"].isin(ADAPTATIONS)]

k_cols = ["K=1", "K=2", "K=5", "K=10", "K=15"]
k_vals = [1, 2, 5, 10, 15]

# Mean fit_time across all models, per adaptation.
means = df.groupby("method")[k_cols].mean()

cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(7, 5))

for i, method in enumerate(ADAPTATIONS):
    ys = means.loc[method]
    xs = [k for k, c in zip(k_vals, k_cols) if pd.notna(ys[c])]
    yv = [ys[c] for c in k_cols if pd.notna(ys[c])]
    ax.plot(xs, yv, marker="o", ms=5, lw=1.8, color=cmap(i),
            label=LABELS.get(method, method))

ax.set_xlabel("K (calibration minutes)")
ax.set_ylabel("Mean fit_time (s)")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fit_time_vs_k.png", dpi=150, bbox_inches="tight")
print(f"wrote {OUT_DIR / 'fit_time_vs_k.png'}")
