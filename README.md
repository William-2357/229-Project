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

| Model | Key Details |
|-------|------------|
| `MIRepNet` | Resamples to 250 Hz, maps to 45-channel template, 256-dim features |
| `NeuroGPT` | Encoder + embedder portions, 22 channels, 250 Hz / 2-second chunks |
| `CBraMod` | Spectral patch embedding, mean-pools to one feature vector |
| `LaBraM` | Channel-index mapping, temporal patching, own feature normalization |

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
