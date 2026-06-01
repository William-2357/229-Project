"""Modal runner for EEG calibration efficiency experiments.

Fans out one GPU job per (method, subject_id) with max 10 concurrent GPUs.

Setup (one-time):
    modal volume create eeg-data
    modal volume put eeg-data data/raw/bciciv2a /bciciv2a

Run:
    modal run modal_runner.py
    modal run modal_runner.py --dataset synthetic --backbone eegnet
    modal run modal_runner.py --dataset bciciv2a --methods loso ea tta
"""

import modal
import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

#DATASET = "bciciv2a"       # bciciv2a | synthetic
#BACKBONE = "eegnet"        # eegnet | shallowconv
#METHODS = ["loso", "ea", "tta", "finetune", "lora", "ea_lora", "cld", "ea_cld"]  # or subset
#K_MINUTES = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]
#N_REPEATS = 5
#SEED = 42
#GPU = "A10G"                 # T4 | A10G | A100
#MAX_CONCURRENCY = 20       # max simultaneous GPUs
#CHECKPOINT_PATH = None     # path to pretrained weights for foundation backbones (e.g. "/data/neurogpt.pt")

DATASET = "bciciv2a"
BACKBONE = "mirepnet"          # mirepnet | neurogpt | cbramod
METHODS = [
    "foundation_sft_loso",     # K=0 zero-shot: source-finetuned backbone, no target adapt
    "foundation_sft_ea",       # K=0 zero-shot: EA + source-finetuned backbone
    "foundation_sft_tta",      # K=0 zero-shot: source-finetuned backbone + T3A
    "foundation_sft_finetune", # K>0: source-finetuned backbone + target finetune
    "foundation_sft_lora",     # K>0: source-finetuned backbone + LoRA
    "foundation_sft_ea_lora",  # K>0: EA + source-finetuned backbone + LoRA
    "foundation_sft_cld",              # K>0: source-finetuned backbone + CLD head
    "foundation_sft_ea_cld",           # K>0: EA + source-finetuned backbone + CLD head
   # "foundation_sft_anchored_cld",     # K>0: source-anchored 2-stage warm ADMM
   # "foundation_sft_ea_anchored_cld",  # K>0: EA + source-anchored 2-stage warm ADMM
]
CHECKPOINT_PATH = "/data/MIRepNet.pth"  # path inside container (mounted from eeg-data volume)
GPU = "A10G"
MAX_CONCURRENCY = 20
K_MINUTES = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]
N_REPEATS = 5
SEED = 42



# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("eeg-experiments")

# Volume stores raw data (npz files); mount at /data inside container
data_volume = modal.Volume.from_name("eeg-data", create_if_missing=True)

# Volume caches JAX XLA compilation artifacts so GPU kernels survive container restarts
jax_cache_volume = modal.Volume.from_name("jax-xla-cache", create_if_missing=True)

# Image: Python 3.11 + all dependencies from requirements.txt
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24,<2.0",
        "scipy>=1.10",
        "mne>=1.6",
        "braindecode>=0.8",
        "torch>=2.0",
        "peft>=0.6",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "tqdm>=4.65",
        "h5py>=3.9",
        "requests>=2.31",
        "jax[cuda12]",
        "jaxcld>=0.1.0",
    )
    .add_local_python_source(
        "adaptation", "data", "evaluation", "models", "plotting",
        copy=True,
    )
)

# ---------------------------------------------------------------------------
# Unsupervised methods (K=0 only): loso, ea, tta
# ---------------------------------------------------------------------------

UNSUPERVISED = {
    "loso", "ea", "tta",
    "linear_probe", "foundation_loso", "foundation_ea", "foundation_tta",
    "foundation_sft_loso", "foundation_sft_ea", "foundation_sft_tta",
}
SUPERVISED = {
    "finetune", "lora", "ea_lora", "cld", "ea_cld",
    "foundation_finetune", "foundation_lora", "foundation_ea_lora",
    "foundation_cld", "foundation_ea_cld",
    "foundation_sft_finetune", "foundation_sft_lora", "foundation_sft_ea_lora",
    "foundation_sft_cld", "foundation_sft_ea_cld",
    "foundation_sft_anchored_cld", "foundation_sft_ea_anchored_cld",
}


# ---------------------------------------------------------------------------
# Remote function — one GPU per (method, subject_id) call
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/data": data_volume, "/root/.cache/jax_xla": jax_cache_volume},

    timeout=7200,
    max_containers=MAX_CONCURRENCY,
)
def run_job(
    method: str,
    subject_id: int,
    dataset_name: str,
    backbone_name: str,
    k_minutes: list,
    n_repeats: int,
    seed: int,
    checkpoint_path: str | None = None,
) -> dict:
    """Run one (method, subject) experiment and return serializable results."""
    import sys
    import copy
    import time
    import numpy as np
    import torch
    import jax
    jax.config.update("jax_platform_name", "gpu")
    jax.config.update("jax_compilation_cache_dir", "/root/.cache/jax_xla")


    from data.datasets import BCICIVDataset
    from data.synthetic import SyntheticDataset
    from models.specialists import build_backbone
    from models.foundations import build_foundation_model, FOUNDATION_NAMES
    from adaptation.loso import LOSOAdapter
    from adaptation.ea import EAAdapter
    from adaptation.tta import TTAAdapter
    from adaptation.finetune import FineTuneAdapter
    from adaptation.lora import LoRAAdapter
    from adaptation.ea_lora import EALoRAAdapter
    from adaptation.cld import CLDAdapter
    from adaptation.stacked import EACLDAdapter
    from adaptation.foundation_cld import FoundationCLDAdapter, FoundationEACLDAdapter
    from adaptation.foundation_source_cld import (
        FoundationSourceFineTuneCLDAdapter, FoundationSourceFineTuneEACLDAdapter
    )
    from adaptation.foundation_finetune import FoundationFineTuneAdapter
    from adaptation.foundation_lora import FoundationLoRAAdapter
    from adaptation.foundation_source_lora import (
        FoundationSourceFineTuneLoRAAdapter, FoundationSourceFineTuneEALoRAAdapter
    )
    from adaptation.foundation_ea import FoundationEAAdapter
    from adaptation.foundation_ea_lora import FoundationEALoRAAdapter
    from adaptation.linear_probe import LinearProbeAdapter
    from adaptation.foundation_loso import FoundationLOSOAdapter
    from adaptation.foundation_tta import FoundationTTAAdapter
    from adaptation.foundation_source_loso import FoundationSFTLOSOAdapter, FoundationSFTEAAdapter
    from adaptation.foundation_source_tta import FoundationSFTTTAAdapter
    from adaptation.foundation_sft_finetune import FoundationSFTFineTuneAdapter
    from adaptation.foundation_sft_anchored_cld import (
        FoundationSFTAnchoredCLDAdapter, FoundationSFTAnchoredEACLDAdapter
    )
    from evaluation.protocols import loso_evaluation, k_minute_sweep
    from evaluation.results import save_result

    METHOD_REGISTRY = {
        "loso": LOSOAdapter,
        "ea": EAAdapter,
        "tta": TTAAdapter,
        "finetune": FineTuneAdapter,
        "lora": LoRAAdapter,
        "ea_lora": EALoRAAdapter,
        "cld": CLDAdapter,
        "ea_cld": EACLDAdapter,
        "linear_probe": LinearProbeAdapter,
        "foundation_loso": FoundationLOSOAdapter,
        "foundation_ea": FoundationEAAdapter,
        "foundation_tta": FoundationTTAAdapter,
        "foundation_finetune": FoundationFineTuneAdapter,
        "foundation_lora": FoundationLoRAAdapter,
        "foundation_ea_lora": FoundationEALoRAAdapter,
        "foundation_cld": FoundationCLDAdapter,
        "foundation_ea_cld": FoundationEACLDAdapter,
        "foundation_sft_loso": FoundationSFTLOSOAdapter,
        "foundation_sft_ea": FoundationSFTEAAdapter,
        "foundation_sft_tta": FoundationSFTTTAAdapter,
        "foundation_sft_finetune": FoundationSFTFineTuneAdapter,
        "foundation_sft_lora": FoundationSourceFineTuneLoRAAdapter,
        "foundation_sft_ea_lora": FoundationSourceFineTuneEALoRAAdapter,
        "foundation_sft_cld": FoundationSourceFineTuneCLDAdapter,
        "foundation_sft_ea_cld": FoundationSourceFineTuneEACLDAdapter,
        "foundation_sft_anchored_cld": FoundationSFTAnchoredCLDAdapter,
        "foundation_sft_ea_anchored_cld": FoundationSFTAnchoredEACLDAdapter,
    }

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{method}] subject={subject_id} device={device}", flush=True)

    # Load dataset — each foundation backbone has its own preprocessing config and
    # cache directory to avoid mixing differently-filtered data.
    if dataset_name == "synthetic":
        dataset = SyntheticDataset(n_subjects=9, seed=seed)
    elif dataset_name == "bciciv2a":
        from data.preprocessing import (
            BACKBONE_PREPROCESS_CONFIGS, BACKBONE_CACHE_SUFFIX, BACKBONE_TARGET_SFREQ,
        )
        preprocess_cfg = BACKBONE_PREPROCESS_CONFIGS.get(backbone_name)
        cache_suffix   = BACKBONE_CACHE_SUFFIX.get(backbone_name)
        target_sfreq   = BACKBONE_TARGET_SFREQ.get(backbone_name, 200.0)
        cache_dir      = f"/data/bciciv2a_{cache_suffix}_cache" if cache_suffix else "/data/bciciv2a_cache"
        dataset = BCICIVDataset(
            "/data/bciciv2a",
            cache_dir=cache_dir,
            target_sfreq=target_sfreq,
            **({"preprocess_config": preprocess_cfg} if preprocess_cfg else {}),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    input_sfreq = getattr(dataset, "target_sfreq", 200.0)
    n_times = int(4.0 * input_sfreq)
    if backbone_name in FOUNDATION_NAMES:
        backbone = build_foundation_model(
            backbone_name,
            n_channels=dataset.n_channels,
            n_times=n_times,
            checkpoint_path=checkpoint_path,
            input_sfreq=input_sfreq,
            freeze=True,
        )
    else:
        backbone = build_backbone(
            backbone_name,
            n_channels=dataset.n_channels,
            n_classes=dataset.n_classes,
            n_times=n_times,
        )

    adapter_class = METHOD_REGISTRY[method]
    common_kwargs = dict(backbone=copy.deepcopy(backbone), device=device, seed=seed)

    output_dir = "/project/results"
    t0 = time.time()
    results = {}

    if method in UNSUPERVISED:
        result = loso_evaluation(
            dataset=dataset,
            subject_id=subject_id,
            adapter_class=lambda **kw: adapter_class(**common_kwargs),
            adapter_kwargs={},
            seed=seed,
        )
        result["k_minutes"] = 0.0
        save_result(result, output_dir, dataset_name, backbone_name, method,
                    subject_id=subject_id, k_minutes=0.0)
        results[0.0] = result
    else:
        source_cache: dict = {}
        per_k_results = k_minute_sweep(
            dataset=dataset,
            subject_id=subject_id,
            adapter_class=lambda seed, **kw: adapter_class(**{**common_kwargs, "seed": seed}),
            adapter_kwargs={},
            k_minutes_list=k_minutes,
            n_repeats=n_repeats,
            seed=seed,
            n_classes=dataset.n_classes,
            epoch_len_sec=4.0,
            source_cache=source_cache,
        )
        for k, repeats in per_k_results.items():
            agg = {
                "bca": float(np.mean([r["bca"] for r in repeats])),
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
            save_result(agg, output_dir, dataset_name, backbone_name, method,
                        subject_id=subject_id, k_minutes=k)
            results[k] = agg

    elapsed = time.time() - t0
    print(f"[{method}] subject={subject_id} done in {elapsed:.1f}s", flush=True)

    # Return serializable summary (strip non-serializable objects)
    return {
        str(k): {kk: vv for kk, vv in v.items() if kk != "repeats"}
        for k, v in results.items()
    }


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates all jobs and downloads results
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={"/project": modal.Volume.from_name("eeg-results", create_if_missing=True)},
    timeout=86400,  # 24h — orchestrator waits for all jobs
)
def orchestrate(
    dataset: str,
    backbone: str,
    method_list: list,
    all_subjects: list,
    k_minutes: list,
    n_repeats: int,
    seed: int,
    checkpoint_path: str | None = None,
) -> dict:
    # Load any previously completed results so restarts are idempotent
    out_path = Path("/project/results") / dataset / backbone / "modal_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
        print(f"Resuming: found {len(all_results)} existing results")
    else:
        all_results = {}

    all_jobs = [
        (method, subject_id)
        for method in method_list
        for subject_id in all_subjects
    ]
    pending_jobs = [(m, s) for m, s in all_jobs if f"{m}/subject_{s:02d}" not in all_results]
    print(f"Dispatching {len(pending_jobs)} jobs ({len(all_jobs) - len(pending_jobs)} skipped as already done)")
    print(f"Dataset: {dataset} | Backbone: {backbone} | GPU: {GPU}")

    if not pending_jobs:
        print("All jobs already completed.")
        return all_results

    for (method, subject_id), result in zip(
        pending_jobs,
        run_job.starmap(
            [
                (method, subj, dataset, backbone, k_minutes, n_repeats, seed, checkpoint_path)
                for method, subj in pending_jobs
            ]
        ),
    ):
        key = f"{method}/subject_{subject_id:02d}"
        all_results[key] = result
        out_path.write_text(json.dumps(all_results, indent=2))  # checkpoint after each job
        print(f"  Finished: {key}", flush=True)

    print(f"\nAll done. Results saved to {out_path}")
    return all_results


@app.local_entrypoint()
def main(
    dataset: str = DATASET,
    backbone: str = BACKBONE,
    methods: str = "",   # comma-separated, empty = all
    n_subjects: str = "all",
    checkpoint_path: str | None = CHECKPOINT_PATH,
    smoke: bool = False,  # quick sanity check: 1 subject, 2 methods, minimal k
):
    method_list = [m.strip() for m in methods.split(",")] if methods else METHODS

    if dataset == "bciciv2a":
        all_subjects = list(range(1, 10))
    elif dataset == "synthetic":
        all_subjects = list(range(1, 6))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if smoke:
        all_subjects = all_subjects[:1]
        method_list = ["loso", "finetune", "lora", "ea_lora"]
        k_minutes = [0.0, 0.5]
        print("SMOKE TEST: 1 subject, 2 methods, k=[0, 0.5]")
    else:
        k_minutes = K_MINUTES

    if n_subjects != "all":
        all_subjects = all_subjects[:int(n_subjects)]

    print(
        f"Starting orchestrator: {len(method_list) * len(all_subjects)} jobs | "
        f"Dataset: {dataset} | Backbone: {backbone} | Checkpoint: {checkpoint_path}"
    )
    orchestrate.remote(dataset, backbone, method_list, all_subjects, k_minutes, N_REPEATS, SEED, checkpoint_path)
