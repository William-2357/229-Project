"""Unpack modal_summary.json → individual result files → summary CSV → plot.

Auto-discovers all modal_summary.json files under OUTPUT_DIR.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.results import save_result, compile_summary_table, results_to_csv, print_summary_table

OUTPUT_DIR = "results"
METHODS = ["loso", "ea", "tta", "finetune", "lora", "ea_lora", "cld", "ea_cld"]
K_MINUTES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]

DATASET_LABELS = {
    "bciciv2a": "BCI Competition IV 2a",
    "synthetic": "Synthetic",
}
BACKBONE_LABELS = {
    "eegnet": "EEGNet",
    "shallowconv": "ShallowConvNet",
    "conformer": "EEG-Conformer",
}

colors_k0 = {"loso": "#1f77b4", "ea": "#ff7f0e", "tta": "#2ca02c"}
colors_adapt = {
    "finetune": "#d62728", "lora": "#9467bd", "ea_lora": "#8c564b",
    "cld": "#e377c2", "ea_cld": "#17becf",
}
markers = {"finetune": "o", "lora": "s", "ea_lora": "^", "cld": "D", "ea_cld": "P"}
labels_map = {
    "loso": "LOSO", "ea": "EA", "tta": "TTA",
    "finetune": "Fine-tune", "lora": "LoRA", "ea_lora": "EA+LoRA",
    "cld": "CLD", "ea_cld": "EA+CLD",
}

summary_files = sorted(Path(OUTPUT_DIR).glob("*/*/modal_summary.json"))
if not summary_files:
    print(f"No modal_summary.json files found under {OUTPUT_DIR}/")
    raise SystemExit(1)

for summary_path in summary_files:
    backbone = summary_path.parent.name
    dataset = summary_path.parent.parent.name

    # ---------------------------------------------------------------------------
    # 1. Unpack modal_summary.json into individual result files
    # ---------------------------------------------------------------------------

    data = json.loads(summary_path.read_text())
    print(f"\n[{dataset}/{backbone}] Unpacking {len(data)} job results...")
    for job_key, k_results in data.items():
        method, subj_str = job_key.split("/")
        subject_id = int(subj_str.split("_")[1])
        for k_str, result in k_results.items():
            k = float(k_str)
            save_result(result, OUTPUT_DIR, dataset, backbone, method,
                        subject_id=subject_id, k_minutes=k)

    # ---------------------------------------------------------------------------
    # 2. Rebuild summary CSV
    # ---------------------------------------------------------------------------

    rows = compile_summary_table(OUTPUT_DIR, dataset, backbone, METHODS, K_MINUTES)
    print_summary_table(rows)

    csv_path = summary_path.parent / "summary.csv"
    results_to_csv(rows, csv_path)
    print(f"Summary CSV: {csv_path}")

    # ---------------------------------------------------------------------------
    # 3. Plot
    # ---------------------------------------------------------------------------

    df = pd.DataFrame(rows)
    df_k0 = df[(df["k_minutes"] == 0.0) & df["mean_bca"].notna()]
    df_adapt = df[(df["k_minutes"] > 0.0) & df["mean_bca"].notna()]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Source-only baselines as horizontal dashed lines
    for _, row in df_k0.iterrows():
        m = row["method"]
        if m not in colors_k0:
            continue
        ax.axhline(row["mean_bca"], color=colors_k0[m], linestyle="--",
                   linewidth=2.0, label=labels_map[m], alpha=0.85)

    # Supervised adaptation learning curves
    for method in ["finetune", "lora", "ea_lora", "cld", "ea_cld"]:
        md = df_adapt[df_adapt["method"] == method].sort_values("k_minutes")
        if md.empty:
            continue
        ax.plot(md["k_minutes"], md["mean_bca"],
                marker=markers[method], markersize=9, linewidth=2.5,
                label=labels_map[method], color=colors_adapt[method])
        ax.fill_between(md["k_minutes"],
                        md["mean_bca"] - md["std_bca"],
                        md["mean_bca"] + md["std_bca"],
                        alpha=0.15, color=colors_adapt[method])

    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1.2, label="Chance (4-class)")
    ax.set_xlabel("Minutes of Target Data Available", fontsize=12, fontweight="bold")
    ax.set_ylabel("Balanced Class Accuracy (BCA)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    ds_label = DATASET_LABELS.get(dataset, dataset)
    bb_label = BACKBONE_LABELS.get(backbone, backbone)
    n_subjects = len({v["subject_id"] for job_vals in data.values() for v in job_vals.values()})
    plt.suptitle(f"{ds_label} — {bb_label} ({n_subjects} subjects)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_png = summary_path.parent / "adapter_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {out_png}")
    plt.close()
