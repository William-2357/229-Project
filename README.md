# EEG Calibration Efficiency Benchmark

Benchmarks how much labeled calibration data EEG decoders need to adapt to a new subject. The main experiment is a per-subject K-minute sweep on BCIC-IV-2a, comparing zero-shot baselines against supervised adaptation methods across both specialist and foundation EEG backbones.

## Overview

The experiment design uses leave-one-subject-out (LOSO) style transfer:
- **Source data**: pooled training trials from all other subjects
- **Target calibration pool**: first 80% of the held-out subject's session-1 trials
- **Target test set**: last 20% of that same session-1 split

Methods are evaluated at K=0 (zero/few-shot) and across K in minutes of labeled calibration data.

## Dataset

**BCIC-IV-2a** — 9 subjects, 22 EEG channels, 4 motor imagery classes, 250 Hz, session 1 only.

Expected raw layout:
```
data/raw/bciciv2a/A01T.npz
data/raw/bciciv2a/A02T.npz
...
```

## Backbone Families

### 1. Specialist Backbones (`models/specialists.py`)
`eegnet`, `shallowconv`, `deep4net`, `conformer` — output logits directly; trained/adapted in a standard supervised way.

### 2. Foundation Backbones (`models/foundations.py`)
All expose `get_features(X)` and `feature_dim` via the `FoundationBackbone` interface.

| Model | Feature dim | Key Details |
|-------|-------------|------------|
| `MIRepNet` | 256 | Maps to 45-channel template; receives native 250 Hz data (no resampling round-trip) |
| `NeuroGPT` | 1024 | Encoder + embedder portions, 22 channels, 250 Hz / 2-second chunks |
| `CBraMod` | — | Spectral patch embedding, mean-pools to one feature vector |
| `LaBraM` | — | Channel-index mapping, temporal patching, own feature normalization |

> **Sampling rates:** `MIRepNet` and `NeuroGPT` now run on native **250 Hz** data — the dataset is delivered at the backbone's native rate rather than resampled down and back up, eliminating a lossy round-trip. `LaBraM` and `CBraMod` run at 200 Hz. Per-backbone target rates live in `BACKBONE_TARGET_SFREQ` (`data/preprocessing.py`); **delete stale caches after changing a rate.**

## Preprocessing

Per-backbone preprocessing configs (`data/preprocessing.py`):

| Backbone | Bandpass | Z-score |
|----------|----------|---------|
| `labram` | 0.1–75 Hz | No |
| `cbramod` | 0.5–75 Hz | No |
| `neurogpt` | 0.5–40 Hz | No |
| `mirepnet` | 4–40 Hz | No |
| Specialists | 4–40 Hz | Yes |

Default pipeline: common average reference → bandpass → notch → resample → epoch extraction → per-trial per-channel z-score.

Preprocessed data is cached to `data/*_cache/` locally or `/data/bciciv2a_cache` on Modal.

## Adaptation Methods

Three-tier architecture:

**Tier 1 — Specialist** adapters (backbone outputs logits directly):

| Method | File | Description |
|--------|------|-------------|
| `loso` | `adaptation/loso.py` | Zero-shot source-trained baseline |
| `ea` | `adaptation/ea.py` | Euclidean Alignment baseline |
| `tta` | `adaptation/tta.py` | TENT (if BatchNorm) or T3A |
| `finetune` | `adaptation/finetune.py` | Full supervised fine-tuning |
| `lora` | `adaptation/lora.py` | LoRA adaptation |
| `ea_lora` | `adaptation/ea_lora.py` | EA + LoRA |
| `cld` | `adaptation/cld.py` | Convex label denoising head (JAX) |
| `ea_cld` | `adaptation/stacked.py` | EA + CLD |

**Tier 2 — Foundation (frozen backbone)** adapters:

| Method | Description |
|--------|-------------|
| `linear_probe` | Canonical linear probe on frozen features |
| `foundation_loso` | Linear-probe K=0 baseline |
| `foundation_ea` | EA before feature extraction, then linear probe |
| `foundation_tta` | T3A on frozen backbone |
| `foundation_finetune` | K=0 probe; K>0 full fine-tune |
| `foundation_lora` | K=0 probe; K>0 LoRA on eligible layers |
| `foundation_ea_lora` | EA + LoRA |
| `foundation_cld` | Frozen features + CLD head |
| `foundation_ea_cld` | EA + frozen features + CLD head |

**Tier 3 — Foundation SFT** adapters (source fine-tuned then frozen):

Stage 1: Fine-tune full backbone + head on pooled source data.
Stage 2: Freeze backbone; apply lightweight target adaptation.

| Method | Description |
|--------|-------------|
| `foundation_sft_loso` | SFT backbone, no target adaptation |
| `foundation_sft_ea` | EA align + SFT backbone (zero-shot) |
| `foundation_sft_tta` | SFT backbone + T3A |
| `foundation_sft_finetune` | SFT backbone + full target fine-tune |
| `foundation_sft_lora` | SFT backbone + LoRA |
| `foundation_sft_ea_lora` | EA + SFT backbone + LoRA |
| `foundation_sft_cld` | SFT backbone + CLD head |
| `foundation_sft_ea_cld` | EA + SFT backbone + CLD head |

SFT hyperparameter defaults (hardcoded in each adapter file):
- `lr_src=1e-3`, `weight_decay=1e-4`, `max_epochs_src=200`, `patience_src=25`

### CLD implementation notes

The CLD methods (`cld`, `ea_cld`, `foundation_cld`, `foundation_sft_cld`, and the
anchored variants) use a **JAX-based** convex ADMM head via the `jaxcld` library.
A few non-obvious details:

- **XLA recompilation fixed via fixed-bucket padding.** The jitted PCG/ADMM solver
  recompiled on every call because the sample count `N` varied across K values
  (and across source vs. target stages), and `jit` retraces on shape changes.
  Features are now zero-padded up to a fixed bucket size (`pad_features_to_bucket`
  in `adaptation/cld.py`) so the solver compiles once and is reused. The padding is
  solution-invariant.
- **Nyström preconditioner pinned to CPU.** On Modal GPUs, `jaxcld`'s Nyström
  preconditioner (`rand_nys_appx`) crashed with `cuSolver INTERNAL` because its
  `qr`/`cholesky`/`solve_triangular`/`svd` are cuSolver routines. These act on
  tiny (PCA-reduced) matrices, so `adaptation/_jaxcld_cpu_linalg.py` monkey-patches
  them onto the CPU while the expensive sketch matvecs and PCG/ADMM matmuls (cuBLAS)
  stay on GPU. Numerically identical to upstream.
- **High-dim backbones are PCA-reduced** before the CLD head (e.g. NeuroGPT's
  1024-dim features) to keep the ADMM weight tensors small enough for XLA.

## Running Experiments

### Local

```bash
python run_experiment.py --dataset bciciv2a --backbone labram \
    --methods foundation_sft_lora foundation_sft_cld \
    --checkpoint-path /path/to/LaBraM.pth
```

### Modal (cloud, GPU)

```bash
modal run modal_runner.py
```

Default Modal config: LaBraM backbone, A10G GPU, 20 concurrent jobs, K-minutes sweep `[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]`.

The orchestrator is restart-friendly — it resumes from an existing `modal_summary.json` and checkpoints after each finished job.

## Results Layout

```
results/
  {dataset}/
    {backbone}/
      {method}/
        subject_01_k0.5.json
        subject_01_k1.0.json
        ...
      summary.csv
      modal_summary.json
```

Current result directories: `eegnet`, `shallowconv`, `conformer` (specialist); `cbramod`, `mirepnet`, `neurogpt`, `labram` (foundation).

## Key Files

| File | Role |
|------|------|
| `run_experiment.py` | Local CLI entrypoint |
| `modal_runner.py` | Modal remote orchestrator |
| `data/datasets.py` | `BCICIVDataset`, `SyntheticDataset` |
| `data/preprocessing.py` | Preprocessing pipeline + backbone configs |
| `models/specialists.py` | Specialist backbone definitions |
| `models/foundations.py` | Foundation backbone definitions |
| `adaptation/base.py` | Adapter base class |
| `adaptation/foundation_source_finetune.py` | Shared SFT building block |
| `evaluation/protocols.py` | `loso_evaluation`, `k_minute_sweep` |
| `evaluation/results.py` | JSON + CSV result writing |

## Dependencies

- PyTorch, PEFT (LoRA)
- JAX + `jaxcld` (for CLD head)
- MNE (EEG preprocessing)
- Modal (cloud execution)
