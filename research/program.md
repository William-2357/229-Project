# Autoresearch Program — Convex Neural Networks for EEG Calibration

> Standing brief for the autonomous research loop. Read this in full at the
> start of **every** iteration. This file is human-authored; the agent must not
> edit it (except to append to the "Open ideas" backlog when it retires/adds one).

## Mission

Push **test balanced accuracy (BCA)** on the BCIC-IV-2a motor-imagery calibration
benchmark by designing convex / convex-reformulated neural network methods for the
low-resource calibration regime.

## Thesis (why convex)

Convex NN training has provable properties that matter most when calibration data is
scarce: **global optimality** (no bad local minima — the K-minute fit is *the*
optimum, not a lucky one), **convergence guarantees** (ADMM/PCG converge regardless of
init), and **margin / solution stability** (small data perturbations → small solution
change). These translate to robustness in the low-K regime where SGD fine-tuning
overfits. Precedent: the CLD convex head (Feng, Tan & Pilanci, ICML 2026) and its strong
low-resource accent-robust LID results on Whisper encoders; CRONOS / CRONOS-AM
(Mishkin et al., NeurIPS 2024) scale convex reformulations to real architectures.

## The single tracked metric

`score = mean test BCA over evaluated subjects, averaged over the K-minute grid`,
with a **tie-break / emphasis on the low-K end** (0.5–2 min) where convexity should win.
Higher is better. `run_local.py` computes it and writes the per-K curve + K* too.
Never compute the metric on data the method was fit on. The calibration set is the
ONLY labeled target data a method may touch; the test split is sacred.

## What you may edit — exactly ONE file

**`adaptation/convex_calib.py`** — the `ConvexCalibAdapter`. Everything you try lives
here: the convex head, the alignment front-end, hyperparameters (`HPARAMS` dict at the
top), feature extraction, regularization, multi-layer alt-min, etc. It must keep the
`BaseAdapter` interface: `fit(source_data, target_unlabeled, target_labeled,
source_cache=...)`, `predict(X)`, `predict_proba(X)`.

## What you must NOT edit (the fixed harness)

`research/run_local.py`, `research/program.md`, `evaluation/*`, `data/*`,
`models/*`, `adaptation/base.py`, and all other `adaptation/*.py`. Changing the harness
invalidates the leaderboard. If you believe the harness is wrong, write the concern in
the journal and stop — do not silently change it.

## Loop protocol (one iteration)

1. Read `research/journal.md` (recent entries) and `research/leaderboard.json`.
2. Form **one** concrete, falsifiable hypothesis grounded in the thesis / idea backlog.
3. Edit `adaptation/convex_calib.py` to implement it. Keep the diff small and reviewable.
4. **Proxy run** (fast signal):
   `python research/run_local.py --proxy --tag <slug>`
   → synthetic-bootstrapped or real data, 4 subjects, low-K subset.
5. Compare proxy `score` to the leaderboard's proxy best.
   - If it does **not** beat proxy best → record the negative result, revert the file to
     the last leaderboard-winning version, commit, continue.
   - If it **beats** proxy best → run the **full sweep**:
     `python research/run_local.py --full --tag <slug>`
     (all subjects, full K grid). This number is the official score.
6. Append a journal entry (template below). Update `leaderboard.json` if full score is a
   new best.
7. **Commit on `auto`** every iteration (win or lose), message:
   `auto: iter <N> — <one-line summary> (proxy=<x> full=<y or -->)`.
   **Do NOT add any `Co-Authored-By` trailer.** Commits are authored by the user only.

## Hard rules

- Deterministic: respect `--seed` (default 42). Don't introduce nondeterminism that
  makes runs incomparable.
- No test leakage. Fit only on source + the K-minute calibration set.
- Fair budget: a method gets the same calibration trials as the baselines; don't sneak
  extra labeled target data.
- If a run errors or NaNs, that's a failed iteration — journal it, revert, continue.
- One idea per iteration. Resist bundling.

## Open ideas backlog (grounded in the two papers)

- **CRONOS two-layer ReLU head**: replace/augment the CLD head with the CRONOS convex
  reformulation; compare global-optimal head vs. current ADMM head at low K.
- **CRONOS-AM alternating minimization**: alternate (convex head solve) ↔ (light
  backbone/feature-map update) for 2–3 rounds — convexify one block at a time.
- **Margin-stability regularization**: tune `beta` (group-lasso) / add explicit margin
  term; test the hypothesis that larger margin → flatter low-K degradation.
- **EA front-end variants**: the current best convex combo is EA-whitening + convex head.
  Try Riemannian alignment, per-class recentering, shrinkage covariance.
- **Feature source**: penultimate vs. earlier layer; multiple layers concatenated;
  PCA/rank reduction before the convex solve (the `rank` knob).
- **Neuron count / rank sweep**: convex head capacity vs. K — does the optimal capacity
  grow with K as theory predicts?
- **Warm-start across K**: reuse the K=0 convex solution as the ADMM init for K>0.
- **Ensemble of convex heads** over seeds/feature subsets — cheap, variance-reducing.

## Notes / environment

- Local: single RTX 5090 (32 GB), CUDA. `jax` runs on GPU; torch on CUDA.
- Backbone for the convex head is a frozen/source-trained **EEGNet** specialist by default
  (cheap, fast) — foundation backbones need checkpoints not present locally.
- Data: `--proxy`/`--full` default to real BCIC-IV-2a via MOABB once cached; until then
  pass `--dataset synthetic` to bootstrap. `run_local.py` picks real data automatically
  if the cache exists.
