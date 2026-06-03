# EEG Calibration Efficiency Benchmark

Benchmarks how much labeled calibration data EEG decoders need to adapt to a new subject. The main experiment is a per-subject K-minute sweep on BCIC-IV-2a, comparing zero-shot baselines against supervised adaptation methods across both specialist and foundation EEG backbones.

## Overview

The experiment design uses leave-one-subject-out (LOSO) style transfer:
- **Source data**: pooled training trials from all other subjects
- **Target calibration pool**: first 80% of the held-out subject's session-1 trials
- **Target test set**: last 20% of that same session-1 split

Methods are evaluated at K=0 (zero/few-shot) and across K in minutes of labeled calibration data.

## Dataset

**BCIC-IV-2a** — 9 subjects, 22 EEG channels, 4 motor imagery classes, 250 Hz, session 1 only (288 trials/subject, 72 per class).

Trials are extracted as **4-second epochs** (`epoch_len_sec=4.0`). The K-minute sweep
(`evaluation/protocols.py:minutes_to_trials`) converts minutes to a balanced trial count as
`n_trials = 15 × K` (15 four-second epochs per minute), sampled stratified across the 4 classes.

> **Calibration ceiling.** The target calibration pool is the first 80% of the session ≈ **230
> trials ≈ 15.3 min** of 4-second epochs. Balanced sampling saturates there, so **K=15 already
> consumes ~97% of the pool and any K ≳ 15.3 (e.g. K=30) reuses the same ~230 trials** — it is
> *not* a true 30-minute condition. Treat K > 15 as the "full calibration pool" plateau for both
> accuracy and `fit_time`.

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
| `anchored_cld` | `adaptation/anchored_cld.py` | Source-anchored 2-stage CLD (low-K fix; specialist analogue of `foundation_sft_anchored_cld`) |
| `ea_anchored_cld` | `adaptation/anchored_cld.py` | EA + source-anchored 2-stage CLD |

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
| `foundation_sft_cld` | SFT backbone + CLD head (Stage 2 on **target calibration only**) |
| `foundation_sft_ea_cld` | EA + SFT backbone + CLD head (target-only) |
| `foundation_sft_anchored_cld` | SFT backbone + **source-anchored** 2-stage CLD (low-K fix) |
| `foundation_sft_ea_anchored_cld` | EA + SFT backbone + source-anchored 2-stage CLD |

SFT hyperparameter defaults (hardcoded in each adapter file):
- `lr_src=1e-3`, `weight_decay=1e-4`, `max_epochs_src=200`, `patience_src=25`

### Source-anchored CLD (low-K fix)

Plain `foundation_sft_cld` fits the convex head on the **target calibration trials
only**, which underperforms at small K (few-shot). The `*_anchored_cld` variants
implement the fix from `reve_kmin_convexnn_v3.ipynb`: a **two-stage** convex solve.

- **Stage 1** — cold ADMM on the source pool; saves the convex-program primal `(u, v)`.
- **Stage 2** (K > 0) — rebuild `CVX_ReLU_MLP` on **source + weighted calibration**
  (calibration rows repeated to approximate a `target_mass` share of the loss), warm-started
  from the Stage-1 primal (same seed → same random hyperplanes), with the ADMM dual `lam`
  reset. The **Stage-1 feature scaler is reused** so the warm-started weights stay valid.
- K = 0 uses the Stage-1 model directly.

**Per-backbone HP tuning.** The two knobs that define Stage 2 (`beta`, `target_mass`)
are grid-searched (`{3e-4,1e-3,3e-3} × {0.15,0.35,0.55}`) on a **leak-free, source-internal
validation split** (a held-out source slice acts as a pseudo-target). HPs are **tuned once
per backbone** (not per fold): a single "primer" job per anchored variant selects and
persists them to the volume (`{backbone}_anchored_{ea_}hp_seed{seed}.json`), and the
remaining folds reuse them. EA and non-EA variants tune separately (EA whitens the feature
space). HPs are tuned on the foundation-feature space, not copied from the REVE notebook.

### `fit_time` semantics (foundation SFT methods)

For the SFT adapters, `fit_time` measures **only the per-K target adaptation** — the timer
starts after the frozen, source-fine-tuned backbone is ready. Source fine-tuning (cached to
disk), backbone loading, and source feature extraction are excluded so methods are comparable.
The frozen backbone, source features, and Stage-1 solve are cached across K within a job.
(Note: results predating this change bundled SFT training/model-reload into `fit_time` and are
**not** directly comparable across methods — e.g. LoRA's K=0.5 included a full source fine-tune.)

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
- **Anchored Stage 2 pads to a `target_mass`-invariant size.** In the source-anchored
  variants, `X_aug = source + repeated-calibration` changes row count with both K and
  `target_mass`, so next-multiple padding still retraced per call (a NeuroGPT EA run
  took ~7 min vs ~2 min). `fit_stage2_anchored` now pads to `n_src·(1+odds)` rounded up
  plus one bucket — a function of `n_src` and `target_mass` only (both constant for a
  sweep) — so Stage 2 compiles once. The EA variant also caches the aligned source.
- **GPU backend forced and verified.** `modal_runner.py` sets `JAX_PLATFORMS=cuda,cpu`
  on the image (cuda default; cpu kept for the Nyström pin) and the worker raises if
  `jax.default_backend()` isn't a GPU — a silent CPU fallback would run the solver
  ~10–20× slower and otherwise look identical.
- **Nyström preconditioner pinned to CPU.** On Modal GPUs, `jaxcld`'s Nyström
  preconditioner (`rand_nys_appx`) crashed with `cuSolver INTERNAL` because its
  `qr`/`cholesky`/`solve_triangular`/`svd` are cuSolver routines. These act on
  tiny (PCA-reduced) matrices, so `adaptation/_jaxcld_cpu_linalg.py` monkey-patches
  them onto the CPU while the expensive sketch matvecs and PCG/ADMM matmuls (cuBLAS)
  stay on GPU. Numerically identical to upstream.
- **PCA reduction is available but disabled by default** (`max_feat_dim=None`).
  When set, high-dim backbones (e.g. NeuroGPT's 1024-dim features) are PCA-reduced
  before the CLD head to keep the ADMM weight tensors small enough for XLA. With it
  off, the CLD head fits on the full feature space; only NeuroGPT (1024 > 256) was
  ever affected — all other backbones (≤256-dim) are bit-identical either way.

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

Default Modal config (set at the top of `modal_runner.py`): **CBraMod** backbone with the full
foundation-SFT method suite, A10G GPU, 20 concurrent jobs (`MAX_CONCURRENCY`), 5 repeats per K
(`N_REPEATS`), K-minutes sweep `[0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0]`. Change `BACKBONE` /
`METHODS` / `CHECKPOINT_PATH` there or via `--backbone` / `--methods` / `--checkpoint-path`.

> Per the **calibration ceiling** above, the K=30 sweep point is redundant with K≈15.3 (same
> ~230 trials); it is kept only for backward-comparability with earlier runs.

The orchestrator is restart-friendly — it resumes from an existing `modal_summary.json` and checkpoints after each finished job.

## Results Layout

At runtime, `save_result` (`evaluation/results.py`) and the Modal orchestrator write per-job
JSON to `{output_dir}/{dataset}/{backbone}/{method}/`, with a per-backbone `modal_summary.json`
checkpointed after each job (on Modal, `output_dir` is `/project/results`):

```
results/                        # runtime output dir (Modal: /project/results)
  {dataset}/
    {backbone}/
      {method}/
        subject_01_k0.5.json
        ...
      summary.csv
      modal_summary.json
```

In this repo, the **curated** results are committed under two top-level folders (split by
backbone family) rather than a single `results/` tree:

- `foundation_results/{cbramod,labram,mirepnet,neurogpt}/` — foundation backbones
- `specialist_results/{eegnet,shallowconv,conformer}/` — specialist backbones

Each backbone folder holds its `modal_summary.json` (the file consumed by `plot_results.py`).

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
