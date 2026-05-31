"""Fixed evaluation harness for the convex-calibration autoresearch loop.

DO NOT EDIT in the research loop (see research/program.md). Defines the single tracked
metric so leaderboard numbers stay comparable across iterations.

Backbone is a source-fine-tuned MIRepNet foundation model (frozen after source-FT),
matching the `foundation_sft_*` baselines. Any registered method can be run through the
identical pipeline so "convex vs LoRA" is a fair, same-machine comparison.

    score        = mean test BCA over all (subject, K>0) cells           [higher better]
    low_k_score  = mean test BCA over the two smallest K                 [tie-break]
    per_k        = mean BCA at each K (the calibration curve)
    k_star       = min K reaching 80% of the per-K ceiling

Usage:
    python research/run_local.py --proxy --tag idea         # convex, 4 subj, low-K
    python research/run_local.py --full  --tag idea         # convex, 9 subj, full K
    python research/run_local.py --proxy --method foundation_sft_lora --tag lora_base
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import BCICIVDataset
from data.synthetic import SyntheticDataset
from data.preprocessing import BACKBONE_PREPROCESS_CONFIGS, BACKBONE_CACHE_SUFFIX
from models.foundations import build_foundation_model, FOUNDATION_NAMES
from evaluation.protocols import k_minute_sweep
from evaluation.metrics import compute_k_star
from adaptation.convex_calib import ConvexCalibAdapter, HPARAMS
from adaptation.foundation_source_lora import FoundationSourceFineTuneLoRAAdapter
from adaptation.foundation_sft_finetune import FoundationSFTFineTuneAdapter

REPO = Path(__file__).resolve().parent.parent
PROXY_K = [0.5, 1.0, 2.0]
FULL_K = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]
CKPT = {"mirepnet": str(REPO / "MIRepNet.pth")}
# Backbone-native sampling rate. MIRepNet was pretrained at 250 Hz; feeding native 250 Hz
# avoids a lossy 250->200->250 round-trip (+0.03-0.05 BCA, verified). Default 200 Hz.
BACKBONE_SFREQ = {"mirepnet": 250.0}

METHODS = {
    "convex_calib": ConvexCalibAdapter,
    "foundation_sft_lora": FoundationSourceFineTuneLoRAAdapter,
    "foundation_sft_finetune": FoundationSFTFineTuneAdapter,
}


def build_dataset(name: str, backbone: str, data_root: Path, seed: int):
    if name == "synthetic":
        return SyntheticDataset(n_subjects=9, seed=seed), "synthetic"
    data_dir = data_root / "bciciv2a"
    cfg = BACKBONE_PREPROCESS_CONFIGS.get(backbone)
    suffix = BACKBONE_CACHE_SUFFIX.get(backbone, "default")
    tsf = BACKBONE_SFREQ.get(backbone, 200.0)
    cache_dir = str(data_root / f"bciciv2a_{suffix}{int(tsf)}_cache")
    return BCICIVDataset(str(data_dir), cache_dir=cache_dir, target_sfreq=tsf,
                         preprocess_config=cfg), "bciciv2a"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(dataset, backbone_name, subjects, k_grid, seed, device, n_repeats, method):
    n_classes = dataset.n_classes
    sfreq = getattr(dataset, "target_sfreq", 200.0)
    n_times = int(4.0 * sfreq)
    adapter_cls = METHODS[method]

    # Build the (frozen) pretrained backbone once; each adapter deep-copies + source-FTs.
    backbone = build_foundation_model(
        backbone_name, n_channels=dataset.n_channels, n_times=n_times,
        checkpoint_path=CKPT.get(backbone_name), input_sfreq=sfreq, freeze=True)

    per_k_bca = {k: [] for k in k_grid}
    for sid in subjects:
        common = dict(backbone=backbone, device=device, seed=seed)
        source_cache: dict = {}
        per_k = k_minute_sweep(
            dataset=dataset, subject_id=sid,
            adapter_class=lambda seed, **kw: adapter_cls(**{**common, "seed": seed}),
            adapter_kwargs={}, k_minutes_list=k_grid, n_repeats=n_repeats,
            seed=seed, n_classes=n_classes, epoch_len_sec=4.0, source_cache=source_cache)
        for k, repeats in per_k.items():
            per_k_bca[k].append(float(np.mean([r["bca"] for r in repeats])))
        print(f"  subject {sid}: "
              f"{ {k: round(float(np.mean(v)),3) for k,v in per_k_bca.items() if v} }", flush=True)
    return per_k_bca


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--proxy", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="convex_calib", choices=list(METHODS))
    ap.add_argument("--backbone", default="mirepnet")
    ap.add_argument("--dataset", default="bciciv2a", choices=["bciciv2a", "synthetic"])
    ap.add_argument("--data_root", default=str(REPO / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_repeats", type=int, default=None)
    args = ap.parse_args()

    mode = "proxy" if args.proxy else "full"
    seed = args.seed
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset, resolved = build_dataset(args.dataset, args.backbone, Path(args.data_root), seed)
    all_subj = dataset.subject_ids
    if args.proxy:
        subjects, k_grid = all_subj[:4], PROXY_K
        n_repeats = args.n_repeats if args.n_repeats is not None else 3
    else:
        subjects, k_grid = all_subj, FULL_K
        n_repeats = args.n_repeats if args.n_repeats is not None else 5

    device = get_device()
    print(f"[{mode}] method={args.method} backbone={args.backbone} dataset={resolved} "
          f"subjects={subjects} K={k_grid} repeats={n_repeats} device={device}", flush=True)
    if args.method == "convex_calib":
        print(f"HPARAMS={json.dumps(HPARAMS)}", flush=True)

    t0 = time.time()
    per_k_bca = evaluate(dataset, args.backbone, subjects, k_grid, seed, device, n_repeats, args.method)
    elapsed = time.time() - t0

    per_k_mean = {float(k): float(np.mean(v)) for k, v in per_k_bca.items()}
    score = float(np.mean(list(per_k_mean.values())))
    low_ks = sorted(per_k_mean)[:2]
    low_k_score = float(np.mean([per_k_mean[k] for k in low_ks]))
    ceiling = max(per_k_mean.values())
    ks_sorted = sorted(per_k_mean)
    k_star = compute_k_star(ks_sorted, [per_k_mean[k] for k in ks_sorted], ceiling=ceiling)

    result = {
        "tag": args.tag, "method": args.method, "mode": mode, "backbone": args.backbone,
        "dataset": resolved, "subjects": list(map(int, subjects)), "k_grid": k_grid,
        "n_repeats": n_repeats, "seed": seed,
        "score": round(score, 4), "low_k_score": round(low_k_score, 4),
        "per_k": {str(k): round(v, 4) for k, v in per_k_mean.items()},
        "k_star": k_star, "ceiling": round(ceiling, 4), "elapsed_sec": round(elapsed, 1),
        "hparams": HPARAMS if args.method == "convex_calib" else None,
    }
    out_dir = REPO / "research" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}__{mode}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 60)
    print(f"RESULT [{mode}] {args.tag}  method={args.method}")
    print(f"  score (mean BCA over K) : {score:.4f}")
    print(f"  low_k_score (2 lowest K): {low_k_score:.4f}")
    print(f"  per_k                   : {result['per_k']}")
    print(f"  k_star (80% ceiling)    : {k_star}")
    print(f"  elapsed                 : {elapsed:.1f}s   saved: {out_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
