"""Build {slug}fit_time_by_k.csv / .md from the padding=false fit-time runs.

Reads results/fittime/<backbone>/modal_summary.json (pulled from the volume's
modal_summary_fittime.json), averages the selected COMPILE-EXCLUDED timing metric
across subjects for each (backbone, method, K), and writes the wide table used by
the fit_time plots. The *_warm metrics = mean over warm repeats (repeat 0 holds
the JAX compile + one-time anchor build and is dropped).

--metric (or FITTIME_METRIC) picks the timing metric; default fit_time_warm.
Use train_fit_time_warm for the pure on-target training time -> writes
train_fit_time_by_k.csv. See scripts/fittime_metrics.py.
"""
import argparse, glob, json, os
from collections import defaultdict

from fittime_metrics import resolve_metric, add_metric_arg

_ap = argparse.ArgumentParser(description=__doc__)
add_metric_arg(_ap)
_ap.add_argument("--results-dir", default="results/fittime",
                 help="dir holding <backbone>/modal_summary.json (default: results/fittime)")
_args = _ap.parse_args()
metric = resolve_metric(_args.metric)

K_MAP = {"0.0": "K=0", "0.5": "K=0.5", "1.0": "K=1", "2.0": "K=2",
         "5.0": "K=5", "10.0": "K=10", "15.0": "K=15", "30.0": "K=30"}
K_COLS = ["K=0", "K=0.5", "K=1", "K=2", "K=5", "K=10", "K=15", "K=30"]

acc = defaultdict(list)
for path in sorted(glob.glob(os.path.join(_args.results_dir, "*/modal_summary.json"))):
    backbone = os.path.basename(os.path.dirname(path))
    data = json.load(open(path))
    for key, k_dict in data.items():
        method = key.split("/")[0]
        for k_str, rec in k_dict.items():
            ft = metric.get(rec)  # selected metric, compile-excluded by default
            if ft is None or k_str not in K_MAP:
                continue
            acc[(backbone, method, K_MAP[k_str])].append(ft)

rows = defaultdict(dict)
for (backbone, method, kcol), vals in acc.items():
    rows[(backbone, method)][kcol] = sum(vals) / len(vals)
ordered = sorted(rows.keys())

csv_path = f"{metric.slug}fit_time_by_k.csv"
md_path = f"{metric.slug}fit_time_by_k.md"

with open(csv_path, "w") as f:
    f.write("backbone,method," + ",".join(K_COLS) + "\n")
    for (backbone, method) in ordered:
        cells = [f"{rows[(backbone, method)][c]:.3f}" if c in rows[(backbone, method)] else ""
                 for c in K_COLS]
        f.write(f"{backbone},{method}," + ",".join(cells) + "\n")

n_subj = max((len(v) for v in acc.values()), default=0)
with open(md_path, "w") as f:
    f.write(f"# Mean {metric.name} (seconds) by K-minutes — padding=false, compile-excluded\n\n")
    f.write(f"Averaged over up to {n_subj} subjects ({metric.name}: warm repeats only).\n\n")
    f.write("| backbone | method | " + " | ".join(K_COLS) + " |\n")
    f.write("|---|---|" + "|".join(["---"] * len(K_COLS)) + "|\n")
    for (backbone, method) in ordered:
        cells = [f"{rows[(backbone, method)][c]:.2f}" if c in rows[(backbone, method)] else "—"
                 for c in K_COLS]
        f.write(f"| {backbone} | {method} | " + " | ".join(cells) + " |\n")

print(f"wrote {csv_path} and {md_path} ({len(ordered)} rows, up to {n_subj} subjects) "
      f"[metric={metric.name}]")
