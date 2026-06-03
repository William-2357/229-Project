"""Build fit_time_by_k.csv / .md from the padding=false fit-time runs.

Reads results/fittime/<backbone>/modal_summary.json (pulled from the volume's
modal_summary_fittime.json), averages the COMPILE-EXCLUDED fit_time_warm across
subjects for each (backbone, method, K), and writes the wide table used by the
fit_time plots. fit_time_warm = mean over warm repeats (repeat 0 holds the JAX
compile + one-time anchor build and is dropped).
"""
import glob, json, os
from collections import defaultdict

K_MAP = {"0.0": "K=0", "0.5": "K=0.5", "1.0": "K=1", "2.0": "K=2",
         "5.0": "K=5", "10.0": "K=10", "15.0": "K=15", "30.0": "K=30"}
K_COLS = ["K=0", "K=0.5", "K=1", "K=2", "K=5", "K=10", "K=15", "K=30"]

acc = defaultdict(list)
for path in sorted(glob.glob("results/fittime/*/modal_summary.json")):
    backbone = os.path.basename(os.path.dirname(path))
    data = json.load(open(path))
    for key, k_dict in data.items():
        method = key.split("/")[0]
        for k_str, rec in k_dict.items():
            ft = rec.get("fit_time_warm", rec.get("fit_time"))  # compile-excluded
            if ft is None or k_str not in K_MAP:
                continue
            acc[(backbone, method, K_MAP[k_str])].append(ft)

rows = defaultdict(dict)
for (backbone, method, kcol), vals in acc.items():
    rows[(backbone, method)][kcol] = sum(vals) / len(vals)
ordered = sorted(rows.keys())

with open("fit_time_by_k.csv", "w") as f:
    f.write("backbone,method," + ",".join(K_COLS) + "\n")
    for (backbone, method) in ordered:
        cells = [f"{rows[(backbone, method)][c]:.3f}" if c in rows[(backbone, method)] else ""
                 for c in K_COLS]
        f.write(f"{backbone},{method}," + ",".join(cells) + "\n")

n_subj = max((len(v) for v in acc.values()), default=0)
with open("fit_time_by_k.md", "w") as f:
    f.write("# Mean fit_time (seconds) by K-minutes — padding=false, compile-excluded\n\n")
    f.write(f"Averaged over up to {n_subj} subjects (fit_time_warm: warm repeats only).\n\n")
    f.write("| backbone | method | " + " | ".join(K_COLS) + " |\n")
    f.write("|---|---|" + "|".join(["---"] * len(K_COLS)) + "|\n")
    for (backbone, method) in ordered:
        cells = [f"{rows[(backbone, method)][c]:.2f}" if c in rows[(backbone, method)] else "—"
                 for c in K_COLS]
        f.write(f"| {backbone} | {method} | " + " | ".join(cells) + " |\n")

print(f"wrote fit_time_by_k.csv and .md ({len(ordered)} rows, up to {n_subj} subjects)")
