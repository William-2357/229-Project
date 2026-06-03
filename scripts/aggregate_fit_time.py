"""Rebuild fit_time_by_k.csv / .md from the raw modal_summary.json files.

Reads every results/{specialist,foundation}/<backbone>/modal_summary.json,
averages fit_time across subjects for each (backbone, method, K), and writes
the wide table used by plot_fit_time.py.
"""

import glob
import json
import os
from collections import defaultdict

# K-string in JSON -> column label / numeric value
K_MAP = {"0.0": "K=0", "0.5": "K=0.5", "1.0": "K=1", "2.0": "K=2",
         "5.0": "K=5", "10.0": "K=10", "15.0": "K=15", "30.0": "K=30"}
K_COLS = ["K=0", "K=0.5", "K=1", "K=2", "K=5", "K=10", "K=15", "K=30"]

# (backbone, method, K_col) -> list of per-subject fit_times
acc = defaultdict(list)

for path in sorted(glob.glob("results/*/*/modal_summary.json")):
    backbone = os.path.basename(os.path.dirname(path))
    data = json.load(open(path))
    for key, k_dict in data.items():
        method = key.split("/")[0]
        for k_str, rec in k_dict.items():
            ft = rec.get("fit_time")
            if ft is None or k_str not in K_MAP:
                continue
            acc[(backbone, method, K_MAP[k_str])].append(ft)

# Collapse to mean per (backbone, method)
rows = defaultdict(dict)  # (backbone, method) -> {K_col: mean}
for (backbone, method, kcol), vals in acc.items():
    rows[(backbone, method)][kcol] = sum(vals) / len(vals)

ordered = sorted(rows.keys())

# --- CSV (2 decimals, blank for missing) ---
with open("fit_time_by_k.csv", "w") as f:
    f.write("backbone,method," + ",".join(K_COLS) + "\n")
    for (backbone, method) in ordered:
        cells = [f"{rows[(backbone, method)][c]:.2f}" if c in rows[(backbone, method)] else ""
                 for c in K_COLS]
        f.write(f"{backbone},{method}," + ",".join(cells) + "\n")

# --- Markdown (1 decimal, em-dash for missing) ---
n_subj = max((len(v) for v in acc.values()), default=0)
with open("fit_time_by_k.md", "w") as f:
    f.write("# Mean fit_time (seconds) by K-minutes\n\n")
    f.write(f"Averaged over up to {n_subj} subjects.\n\n")
    f.write("| backbone | method | " + " | ".join(K_COLS) + " |\n")
    f.write("|---|---|" + "|".join(["---"] * len(K_COLS)) + "|\n")
    for (backbone, method) in ordered:
        cells = [f"{rows[(backbone, method)][c]:.1f}" if c in rows[(backbone, method)] else "—"
                 for c in K_COLS]
        f.write(f"| {backbone} | {method} | " + " | ".join(cells) + " |\n")

print(f"wrote fit_time_by_k.csv and .md ({len(ordered)} backbone/method rows, "
      f"up to {n_subj} subjects)")
