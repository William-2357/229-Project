"""EEG Calibration Efficiency Benchmark — Experiment Runner CLI.

Usage:
    python run_experiment.py \\
      --dataset jeong2020 \\
      --backbone eegnet \\
      --methods all \\
      --k_minutes 0.5 1 2 5 10 15 30 \\
      --n_subjects all \\
      --seed 42 \\
      --output_dir results/
"""

import argparse
import time
import json
import sys
import copy
import numpy as np
import torch
from pathlib import Path

from data.datasets import JeongDataset, BCICIVDataset
from data.synthetic import SyntheticDataset
from models.specialists import build_backbone
from adaptation.loso import LOSOAdapter
from adaptation.ea import EAAdapter
from adaptation.tta import TTAAdapter
from adaptation.finetune import FineTuneAdapter
from adaptation.lora import LoRAAdapter
from adaptation.ea_lora import EALoRAAdapter
from adaptation.cld import CLDAdapter
from adaptation.stacked import EACLDAdapter
from evaluation.protocols import (
    within_subject_cv, loso_evaluation, k_minute_sweep, aggregate_across_subjects
)
from evaluation.results import save_result, compile_summary_table, print_summary_table, results_to_csv
from evaluation.metrics import compute_k_star

DATASET_REGISTRY = {
    "jeong2020": JeongDataset,
    "bciciv2a": BCICIVDataset,
    "synthetic": SyntheticDataset,
}

METHOD_REGISTRY = {
    "loso": LOSOAdapter,
    "ea": EAAdapter,
    "tta": TTAAdapter,
    "finetune": FineTuneAdapter,
    "lora": LoRAAdapter,
    "ea_lora": EALoRAAdapter,
    "cld": CLDAdapter,
    "ea_cld": EACLDAdapter,
}

ALL_METHODS = list(METHOD_REGISTRY.keys())
K_MINUTES_DEFAULT = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]

# Methods that require labeled calibration (K > 0)
SUPERVISED_METHODS = {"finetune", "lora", "ea_lora", "cld", "ea_cld"}
# Methods that are K=0 only
UNSUPERVISED_METHODS = {"loso", "ea", "tta"}


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_method_on_subject(
    method_name: str,
    adapter_class,
    dataset,
    subject_id: int,
    backbone: torch.nn.Module,
    k_minutes_list: list[float],
    seed: int,
    device: str,
    output_dir: str,
    dataset_name: str,
    backbone_name: str,
    n_repeats: int = 5,
) -> dict:
    """Run one method on one subject across all K values. Returns summary."""
    n_classes = dataset.n_classes
    epoch_len = 4.0

    common_kwargs = dict(backbone=copy.deepcopy(backbone), device=device, seed=seed)

    print(f"  [{method_name}] Subject {subject_id} ...", flush=True)
    t0 = time.time()

    if method_name in UNSUPERVISED_METHODS:
        # Single evaluation: train on source, eval on target (K=0)
        result = loso_evaluation(
            dataset=dataset,
            subject_id=subject_id,
            adapter_class=lambda **kw: adapter_class(**common_kwargs),
            adapter_kwargs={},
            seed=seed,
        )
        # Wrap in per-K format for consistency
        result["k_minutes"] = 0.0
        save_result(
            result, output_dir, dataset_name, backbone_name, method_name,
            subject_id=subject_id, k_minutes=0.0,
        )
        summary = {0.0: result}

    else:
        # K-minute sweep; source_cache avoids re-training the same source model
        # for each K value (reused per unique seed/repeat, 7x fewer source trainings)
        # Original (no caching): k_minute_sweep(...) without source_cache
        source_cache: dict = {}
        per_k_results = k_minute_sweep(
            dataset=dataset,
            subject_id=subject_id,
            adapter_class=lambda seed, **kw: adapter_class(**{**common_kwargs, "seed": seed}),
            adapter_kwargs={},
            k_minutes_list=k_minutes_list,
            n_repeats=n_repeats,
            seed=seed,
            n_classes=n_classes,
            epoch_len_sec=epoch_len,
            source_cache=source_cache,
        )
        summary = {}
        for k, repeats in per_k_results.items():
            # Average across repeats and save
            mean_bca = float(np.mean([r["bca"] for r in repeats]))
            agg = {
                "bca": mean_bca,
                "std_bca": float(np.std([r["bca"] for r in repeats])),
                "kappa": float(np.mean([r["kappa"] for r in repeats])),
                "ci_lo": float(np.mean([r["ci_lo"] for r in repeats])),
                "ci_hi": float(np.mean([r["ci_hi"] for r in repeats])),
                "fit_time": float(np.mean([r["fit_time"] for r in repeats])),
                "k_minutes": float(k),
                "n_cal_trials": repeats[0]["n_cal_trials"],
                "protocol": "k_minute_sweep",
                "subject_id": int(subject_id),
                "repeats": repeats,
            }
            save_result(
                agg, output_dir, dataset_name, backbone_name, method_name,
                subject_id=subject_id, k_minutes=k,
            )
            summary[k] = agg

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s", flush=True)
    return summary


def compute_and_print_summary(
    output_dir: str,
    dataset_name: str,
    backbone_name: str,
    methods: list[str],
    k_minutes_list: list[float],
) -> None:
    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY — {dataset_name} / {backbone_name}")
    print("=" * 70)
    rows = compile_summary_table(output_dir, dataset_name, backbone_name, methods, k_minutes_list)
    print_summary_table(rows)

    # Save as CSV
    csv_path = Path(output_dir) / dataset_name / backbone_name / "summary.csv"
    results_to_csv(rows, csv_path)
    print(f"\nSummary saved to: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG Calibration Efficiency Benchmark")
    p.add_argument("--dataset", choices=list(DATASET_REGISTRY), default="jeong2020",
                   help="Dataset to use. 'synthetic' requires no data download.")
    p.add_argument("--backbone", choices=["eegnet", "shallowconv", "deep4net", "conformer"], default="eegnet")
    p.add_argument("--methods", nargs="+", default=["all"],
                   help="Methods to run, or 'all'")
    p.add_argument("--k_minutes", nargs="+", type=float, default=K_MINUTES_DEFAULT)
    p.add_argument("--n_subjects", default="all",
                   help="Number of subjects to use (int or 'all')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--data_dir", default="data/raw/",
                   help="Root directory for raw data files")
    p.add_argument("--device", default=None,
                   help="Device to use (cpu/cuda/mps). Auto-detected if not set.")
    p.add_argument("--n_repeats", type=int, default=5,
                   help="Repeats per K value for variance estimation")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = args.device or get_device(args.cpu)
    print(f"Using device: {device}")

    # Resolve methods
    methods = ALL_METHODS if args.methods == ["all"] else args.methods
    for m in methods:
        if m not in METHOD_REGISTRY:
            print(f"ERROR: Unknown method '{m}'. Choose from: {ALL_METHODS}")
            sys.exit(1)

    # Load dataset
    dataset_cls = DATASET_REGISTRY[args.dataset]
    if args.dataset == "synthetic":
        n_subj = 5 if args.n_subjects == "all" else int(args.n_subjects)
        dataset = dataset_cls(n_subjects=n_subj, seed=args.seed)
    else:
        data_dir = Path(args.data_dir) / args.dataset
        dataset = dataset_cls(str(data_dir))
    print(f"Dataset: {args.dataset} — {dataset.n_channels}ch, {dataset.n_classes} classes")

    # Resolve subjects
    all_subj = dataset.subject_ids
    if args.n_subjects == "all":
        subjects = all_subj
    else:
        subjects = all_subj[:int(args.n_subjects)]
    print(f"Running {len(subjects)} subjects: {subjects}")

    # Determine n_times for backbone construction
    # 4-second epochs at 200 Hz = 800 samples
    n_times = int(4.0 * 200)

    print(f"\nBackbone: {args.backbone}")
    print(f"Methods: {methods}")
    print(f"K values: {args.k_minutes}")
    print(f"Output: {args.output_dir}")
    print()

    # Per-subject loop
    per_method_per_subject: dict[str, list] = {m: [] for m in methods}

    for subject_id in subjects:
        print(f"\nSubject {subject_id}/{len(subjects)}")
        # Build fresh backbone per subject (avoid cross-contamination)
        backbone = build_backbone(
            args.backbone,
            n_channels=dataset.n_channels,
            n_classes=dataset.n_classes,
            n_times=n_times,
        )

        for method_name in methods:
            adapter_class = METHOD_REGISTRY[method_name]
            try:
                summary = run_method_on_subject(
                    method_name=method_name,
                    adapter_class=adapter_class,
                    dataset=dataset,
                    subject_id=subject_id,
                    backbone=backbone,
                    k_minutes_list=args.k_minutes,
                    seed=args.seed,
                    device=device,
                    output_dir=args.output_dir,
                    dataset_name=args.dataset,
                    backbone_name=args.backbone,
                    n_repeats=args.n_repeats,
                )
                # Collect for cross-subject aggregation (use last K or K=0)
                last_k = max(summary.keys())
                per_method_per_subject[method_name].append(summary[last_k])
            except Exception as e:
                print(f"  ERROR [{method_name}] subject {subject_id}: {e}")
                import traceback
                traceback.print_exc()

    compute_and_print_summary(
        args.output_dir, args.dataset, args.backbone, methods, [0.0] + list(args.k_minutes)
    )


if __name__ == "__main__":
    main()
