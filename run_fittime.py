"""Local fit-time benchmark — the in-process twin of scripts/run_fittime_modal.sh.

Runs the K-spanning adaptation methods across the 7 backbones (3 subjects each)
on THIS machine instead of Modal, padding=false for the convex (CLD) solves
(CLD_NO_PAD=1), and writes per-backbone

    <output-root>/<backbone>/modal_summary<tag>.json

in the SAME shape modal_runner produces, so scripts/aggregate_fittime.py and the
fit-time plots consume it unchanged. Each (method, K) record carries
fit_time / fit_time_warm and train_fit_time / train_fit_time_warm (the warm
variants drop repeat 0, which holds the JAX/XLA compile + one-time anchor build).

Method sets mirror the Modal shell:
  foundation backbones (cbramod/labram/mirepnet/neurogpt):  foundation_sft_* methods
  specialist backbones (eegnet/shallowconv/conformer):      bare-name methods

Run (use the project venv, which has torch/jax/peft):
    .venv/bin/python run_fittime.py                          # all 7 backbones, 3 subjects
    .venv/bin/python run_fittime.py --backbones eegnet       # one specialist (no checkpoint needed)
    .venv/bin/python run_fittime.py --backbones cbramod --checkpoint-dir ~/checkpoints
    .venv/bin/python run_fittime.py --n-subjects 1 --k-minutes 0.5 1 2   # quick smoke

Then aggregate + plot the pure-training metric:
    python scripts/aggregate_fittime.py --metric train_fit_time_warm
    python scripts/plot_fittime_final.py --metric train_fit_time_warm

Note: the convex methods (cld/anchored/kadaptive) run their ADMM solver on JAX.
The adapters request the GPU backend; on a CUDA box this is automatic. On a
machine without a JAX GPU, export JAX_PLATFORMS=cpu before running (slower).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# CLD_NO_PAD must be set before the convex solve runs (pad_features_to_bucket reads
# it per call); set it at import time so --no-pad (the default) takes effect for the
# whole process, matching run_fittime_modal.sh's `export CLD_NO_PAD=1`.
_PAD_ENV_PRESET = os.environ.get("CLD_NO_PAD")


# ---------------------------------------------------------------------------
# Config — mirrors scripts/run_fittime_modal.sh
# ---------------------------------------------------------------------------

FOUNDATION_BACKBONES = ["cbramod", "labram", "mirepnet", "neurogpt"]
SPECIALIST_BACKBONES = ["eegnet", "shallowconv", "conformer"]
ALL_BACKBONES = FOUNDATION_BACKBONES + SPECIALIST_BACKBONES

# Foundation checkpoint filenames (joined with --checkpoint-dir).
CHECKPOINT_FILES = {
    "cbramod": "CBraMod_checkpoint.pth",
    "labram": "labram-base.pth",
    "mirepnet": "MIRepNet.pth",
    "neurogpt": "neuro_gpt.pt",
}

# Method sets per backbone family (identical to the Modal shell).
FOUNDATION_METHODS = [
    "foundation_sft_finetune", "foundation_sft_lora", "foundation_sft_ea_lora",
    "foundation_sft_cld", "foundation_sft_ea_cld",
    "foundation_sft_kadaptive_anchored_cld", "foundation_sft_ea_kadaptive_anchored_cld",
]
SPECIALIST_METHODS = [
    "finetune", "lora", "ea_lora", "cld", "ea_cld",
    "kadaptive_anchored_cld", "ea_kadaptive_anchored_cld",
]

K_MINUTES_DEFAULT = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]


# ---------------------------------------------------------------------------
# Dataset loading — mirrors run_experiment.main() so local caches line up
# ---------------------------------------------------------------------------

def load_dataset(RE, dataset_name: str, backbone: str, data_dir: str, seed: int, n_subjects):
    """Build the dataset with the backbone-specific preprocessing config."""
    dataset_cls = RE.DATASET_REGISTRY[dataset_name]
    if dataset_name == "synthetic":
        n_subj = 5 if n_subjects == "all" else int(n_subjects)
        return dataset_cls(n_subjects=n_subj, seed=seed)

    ddir = Path(data_dir) / dataset_name
    from data.preprocessing import (
        BACKBONE_PREPROCESS_CONFIGS, BACKBONE_CACHE_SUFFIX, BACKBONE_TARGET_SFREQ,
    )
    preprocess_cfg = BACKBONE_PREPROCESS_CONFIGS.get(backbone)
    cache_suffix = BACKBONE_CACHE_SUFFIX.get(backbone)
    target_sfreq = BACKBONE_TARGET_SFREQ.get(backbone, 200.0)
    cache_dir = str(ddir.parent / f"{dataset_name}_{cache_suffix}_cache") if cache_suffix else None
    return dataset_cls(
        str(ddir),
        target_sfreq=target_sfreq,
        preprocess_config=preprocess_cfg,
        **({"cache_dir": cache_dir} if cache_dir else {}),
    )


# ---------------------------------------------------------------------------
# Per-backbone run
# ---------------------------------------------------------------------------

def run_backbone(RE, backbone, methods, checkpoint_path, args):
    """Run all methods x subjects for one backbone; write modal_summary.json.

    Mirrors modal_runner.orchestrate + run_job locally: builds the backbone once
    per subject, runs each method via run_experiment.run_method_on_subject (which
    sweeps K and aggregates fit_time/fit_time_warm/train_fit_time[_warm]), and
    accumulates the modal_summary.json structure
        {"<method>/subject_XX": {"<K>": {agg without repeats}, ...}, ...}
    Idempotent: an existing summary file is loaded and finished keys are skipped.
    """
    import copy
    from models.foundations import FOUNDATION_NAMES

    out_dir = Path(args.output_root) / backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"modal_summary{args.tag}.json"

    all_results: dict = {}
    if out_path.exists() and not args.fresh:
        all_results = json.loads(out_path.read_text())
        print(f"  resuming: {len(all_results)} (method/subject) results already present")

    dataset = load_dataset(RE, args.dataset, backbone, args.data_dir, args.seed, args.n_subjects)
    all_subj = dataset.subject_ids
    subjects = all_subj if args.n_subjects == "all" else all_subj[: int(args.n_subjects)]
    print(f"  dataset={args.dataset} backbone={backbone} "
          f"{dataset.n_channels}ch/{dataset.n_classes}cls subjects={subjects}")

    input_sfreq = getattr(dataset, "target_sfreq", 200.0)
    n_times = int(4.0 * input_sfreq)

    for subject_id in subjects:
        pending = [m for m in methods
                   if f"{m}/subject_{subject_id:02d}" not in all_results]
        if not pending:
            continue
        # Fresh backbone per subject (foundation needs the checkpoint); each method
        # deepcopies it internally, so methods don't cross-contaminate.
        backbone_model = RE.build_any_backbone(
            backbone,
            n_channels=dataset.n_channels,
            n_classes=dataset.n_classes,
            n_times=n_times,
            checkpoint_path=checkpoint_path if backbone in FOUNDATION_NAMES else None,
            input_sfreq=input_sfreq,
        )
        for method in pending:
            key = f"{method}/subject_{subject_id:02d}"
            print(f"    [{backbone}] {key} ...", flush=True)
            try:
                summary = RE.run_method_on_subject(
                    method_name=method,
                    adapter_class=RE.METHOD_REGISTRY[method],
                    dataset=dataset,
                    subject_id=subject_id,
                    backbone=backbone_model,
                    k_minutes_list=args.k_minutes,
                    seed=args.seed,
                    device=args.device,
                    output_dir=args.save_dir,
                    dataset_name=args.dataset,
                    backbone_name=backbone,
                    n_repeats=args.n_repeats,
                )
            except Exception as e:  # one bad method shouldn't sink the whole run
                import traceback
                print(f"    ERROR [{backbone}] {key}: {e}")
                traceback.print_exc()
                continue
            # Strip non-summary fields (repeats) → modal_summary.json shape.
            all_results[key] = {
                str(k): {kk: vv for kk, vv in agg.items() if kk != "repeats"}
                for k, agg in summary.items()
            }
            out_path.write_text(json.dumps(all_results, indent=2))  # checkpoint each
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"  wrote {out_path} ({len(all_results)} method/subject results)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local fit-time benchmark (twin of run_fittime_modal.sh)")
    p.add_argument("--backbones", default="all",
                   help="comma-separated backbones or 'all' (default: all 7)")
    p.add_argument("--methods", default="",
                   help="comma-separated method override; empty = auto by backbone family")
    p.add_argument("--dataset", default="bciciv2a", choices=["bciciv2a", "synthetic"])
    p.add_argument("--data-dir", default="data/raw", help="root dir of raw datasets")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="dir holding the foundation checkpoints (see CHECKPOINT_FILES)")
    p.add_argument("--output-root", default="results/fittime",
                   help="modal_summary.json is written to <output-root>/<backbone>/")
    p.add_argument("--save-dir", default="results/fittime_runs",
                   help="per-(method,subject,K) result files from save_result land here")
    p.add_argument("--n-subjects", default="3", help="int or 'all' (default 3, as in the shell)")
    p.add_argument("--n-repeats", type=int, default=5, help="repeats per K (needed for *_warm)")
    p.add_argument("--k-minutes", nargs="+", type=float, default=K_MINUTES_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu/cuda/mps; auto-detected if unset")
    pad = p.add_mutually_exclusive_group()
    pad.add_argument("--no-pad", dest="no_pad", action="store_true", default=True,
                     help="padding=false for convex solves (CLD_NO_PAD=1) — DEFAULT, matches the shell")
    pad.add_argument("--pad", dest="no_pad", action="store_false",
                     help="keep 256-bucket padding for convex solves")
    p.add_argument("--tag", default="", help="suffix for the summary filename (modal_summary<tag>.json)")
    p.add_argument("--fresh", action="store_true", help="ignore any existing summary (no resume)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # padding knob — set BEFORE importing the adapters (via run_experiment).
    if args.no_pad:
        os.environ["CLD_NO_PAD"] = "1"
    elif _PAD_ENV_PRESET is None:
        os.environ.pop("CLD_NO_PAD", None)

    import run_experiment as RE  # imports torch/jax/peft + all adapters

    args.device = args.device or RE.get_device()
    RE.seed_everything(args.seed)

    backbones = ALL_BACKBONES if args.backbones == "all" else \
        [b.strip() for b in args.backbones.split(",") if b.strip()]
    unknown = [b for b in backbones if b not in ALL_BACKBONES]
    if unknown:
        raise SystemExit(f"unknown backbone(s) {unknown}; choose from {ALL_BACKBONES}")

    method_override = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None

    ckpt_dir = Path(args.checkpoint_dir)
    print(f"Local fit-time benchmark | device={args.device} | CLD_NO_PAD={os.environ.get('CLD_NO_PAD','0')} "
          f"| K={args.k_minutes} | repeats={args.n_repeats} | n_subjects={args.n_subjects}")

    written = []
    for backbone in backbones:
        is_foundation = backbone in FOUNDATION_BACKBONES
        methods = method_override or (FOUNDATION_METHODS if is_foundation else SPECIALIST_METHODS)
        checkpoint_path = None
        if is_foundation:
            checkpoint_path = str(ckpt_dir / CHECKPOINT_FILES[backbone])
            if not os.path.exists(checkpoint_path):
                print(f"!! skipping {backbone}: checkpoint not found at {checkpoint_path} "
                      f"(set --checkpoint-dir)")
                continue
        print(f"\n=== {backbone} ({'foundation' if is_foundation else 'specialist'}) "
              f"| {len(methods)} methods ===")
        t0 = time.time()
        written.append(run_backbone(RE, backbone, methods, checkpoint_path, args))
        print(f"=== {backbone} done in {time.time() - t0:.1f}s ===")

    print("\nALL FIT-TIME RUNS DONE")
    for p in written:
        print(f"  {p}")
    print("\nNext: python scripts/aggregate_fittime.py --metric train_fit_time_warm  (then the plots)")


if __name__ == "__main__":
    main()
