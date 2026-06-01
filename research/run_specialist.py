"""Quick test: my convex methods on the eegnet specialist backbone.

Runs SpecialistConvexAdapter modes (convex = source∪cal head; ft_convex = LoRA-adapt on
cal + source∪cal head) on eegnet, sharing the source-trained backbone across modes. Overlays
the existing eegnet baselines (old cld / lora / finetune) from results/ for comparison.
Default preprocessing (200Hz), 9 subjects.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import BCICIVDataset
from models.specialists import build_backbone
from adaptation.convex_calib_specialist import SpecialistConvexAdapter
from evaluation.protocols import k_minute_sweep

REPO = Path(__file__).resolve().parent.parent
BACKBONE = "eegnet"
K_GRID = [1.0, 10.0, 30.0]
N_REPEATS = 2
SEED = 42


def existing_curve(method):
    ys = []
    for k in K_GRID:
        vals = []
        for s in range(1, 10):
            f = REPO / f"results/bciciv2a/{BACKBONE}/{method}/subject_{s:02d}_k{k}.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["bca"])
        ys.append(np.mean(vals) if vals else np.nan)
    return ys


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCICIVDataset(str(REPO / "data/raw/bciciv2a"), cache_dir=str(REPO / "data/raw/bciciv2a_cache"))
    subjects = ds.subject_ids
    n_times = int(4.0 * ds.target_sfreq)
    bb = build_backbone(BACKBONE, n_channels=ds.n_channels, n_classes=ds.n_classes, n_times=n_times)

    modes = ["convex", "ft_convex", "ft_linear"]
    per_k = {m: {k: [] for k in K_GRID} for m in modes}
    for sid in subjects:
        sc = {}  # shared source-train cache across modes (same seed -> trained once)
        for mode in modes:
            res = k_minute_sweep(
                dataset=ds, subject_id=sid,
                adapter_class=lambda seed, _m=mode, **kw: SpecialistConvexAdapter(
                    backbone=bb, device=dev, seed=seed, mode=_m),
                adapter_kwargs={}, k_minutes_list=K_GRID, n_repeats=N_REPEATS,
                seed=SEED, n_classes=ds.n_classes, epoch_len_sec=4.0, source_cache=sc)
            for k, reps in res.items():
                per_k[mode][k].append(float(np.mean([r["bca"] for r in reps])))
        print(f"subject {sid}: " + " | ".join(
            f"{m}={ {k: round(float(np.mean(v)),3) for k,v in per_k[m].items() if v} }" for m in modes),
            flush=True)

    out = {m: {str(k): round(float(np.mean(v)), 4) for k, v in per_k[m].items()} for m in modes}
    for m in modes:
        out[m]["mean"] = round(float(np.mean([np.mean(per_k[m][k]) for k in K_GRID])), 4)
    (REPO / "research/runs/specialist_eegnet_convex.json").write_text(json.dumps(out, indent=2))

    # plot vs existing baselines
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5.4))
    # ours (solid) + same-pipeline linear baseline (solid orange)
    series = [
        ("ft_convex", "tab:green",  "D", "FT-adapt + convex (ours)"),
        ("convex",    "tab:red",    "o", "convex source∪cal (ours)"),
        ("ft_linear", "tab:orange", "s", "FT-adapt + linear = finetune (same pipeline)"),
    ]
    for m, c, mk, lab in series:
        ys = [np.mean(per_k[m][k]) for k in K_GRID]
        plt.plot(K_GRID, ys, marker=mk, color=c, lw=2.2, label=f"{lab} ({np.mean(ys):.3f})")
    # existing modal baselines (dashed, DIFFERENT source-train — context only)
    for method, c, mk, lab in [("lora","tab:gray","s","lora (existing modal)"),
                               ("finetune","lightblue","^","finetune (existing modal)")]:
        ys = existing_curve(method)
        plt.plot(K_GRID, ys, ":", marker=mk, color=c, lw=1.2, alpha=0.7, label=f"{lab} ({np.nanmean(ys):.3f})")
    plt.xscale("log"); plt.xticks(K_GRID, [str(k) for k in K_GRID])
    plt.xlabel("Calibration window K (minutes)"); plt.ylabel("Mean test BCA (9 subjects)")
    plt.title("eegnet specialist — my convex methods (solid) vs existing baselines (dashed)")
    plt.grid(True, alpha=0.3, which="both"); plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); out_png = REPO / "research/specialist_eegnet_convex.png"
    plt.savefig(out_png, dpi=140)
    print("\nRESULT eegnet:", json.dumps(out))
    print("saved", out_png)


if __name__ == "__main__":
    main()
