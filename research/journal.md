# Autoresearch Journal — Convex EEG Calibration

Append-only. Newest entries at the bottom. One entry per iteration.

Template:
```
## iter N — <slug>   (YYYY-MM-DD)
- hypothesis: <one falsifiable sentence, grounded in the thesis/backlog>
- change: <what changed in adaptation/convex_calib.py>
- proxy: score=<x> low_k=<x> per_k=<...>   (best-so-far proxy=<y>)
- decision: KEPT (ran full) | REVERTED (did not beat proxy best) | ERROR
- full: score=<x> low_k=<x> per_k=<...>     (only if promoted)
- notes: <surprises, next idea>
```

---

## iter 0 — baseline (2026-05-31)
- hypothesis: the EA-whitening + convex ADMM ReLU head (EACLD, validated in-repo) is
  the strongest convex starting point for low-K calibration; establish its score as the
  leaderboard floor.
- change: created `adaptation/convex_calib.py` (ConvexCalibAdapter) = EA front-end +
  jaxcld CVX_ReLU_MLP head, single HPARAMS surface. Created fixed harness
  `research/run_local.py` and this journal.
- decision: BASELINE (see leaderboard.json once the bootstrap run lands)
- notes: bootstrap/validate on synthetic first; cut over to real BCIC-IV-2a (MOABB)
  for the official leaderboard. First real idea to try next: CRONOS two-layer ReLU head
  vs. the current ADMM head at low K.

## PIVOT (2026-05-31) — fair FM comparison on source-pretrained MIRepNet
Per user direction + teammate's `bf55826`:
- **Backbone is now MIRepNet** (foundation model), **source-fine-tuned** then frozen —
  the same protocol as the `foundation_sft_*` baselines, so the comparison is fair
  (only the adaptation head differs). EEGNet specialist regime retired.
- **Benchmark target: beat `foundation_sft_lora`** (the bar in mirepnet_performance.png).
  All-subject curve to beat: K=.5→.428 1→.432 2→.423 5→.457 10→.480 15→.528 30→.518.
- **Reconstructed missing infra**: teammate gitignored `foundation_source_finetune.py`,
  `foundation_source_lora.py`, `foundation_source_cld.py` (imports survive but files
  absent). Rebuilt `foundation_source_finetune.py` (shared, disk-cached source-FT) and
  `foundation_source_lora.py` (sft_lora baseline) so baselines + convex run locally on
  the SAME source-FT backbone.
- Harness retargeted: `run_local.py` now builds source-FT MIRepNet, `--method` selects
  convex_calib / foundation_sft_lora / foundation_sft_finetune.

## iter 1 — combined_convex (2026-05-31)
- hypothesis: the `sft_cld` low-K dip happens because the convex head is REFIT FROM
  SCRATCH on ~12 calibration trials at K>0, discarding source knowledge. Fitting the
  convex head on the UNION of source ∪ upweighted-calibration features (one global ADMM
  solve, well-posed at every K) removes the dip and should track/beat LoRA.
- change: rewrote `convex_calib.py` → source-FT MIRepNet (frozen) + convex head fit on
  source ∪ calibration (knobs: source_cap=800, cal_balance=1.0).
- smoke (3-epoch FT, subj1): LOSO=.350 k1=.435 k5=.457 — dip already gone (k1 > LOSO).
- proxy (full source-FT, subj1-4): score=0.5755 low_k=0.569 per_k={.5:.574, 1:.564, 2:.588}.
  Dip GONE (k=.5 already 0.574 vs stock sft_cld 0.357).
- FAIR same-backbone baselines (subj1-4, identical cached source-FT MIRepNet):
    convex      0.5755  (.574/.564/.588)
    sft_finetune 0.5619 (.556/.567/.562)
    sft_lora     0.5570 (.555/.560/.556)
  → convex beats LoRA +0.0185, finetune +0.0136. Biggest edge at low K.
- IMPORTANT: the modal-summary bar was UNFAIR — my source-FT backbone is much stronger
  than the teammate's (my local LoRA 0.557 vs modal LoRA 0.462). The honest claim is the
  same-backbone comparison above: convex > LoRA by ~0.02 on the proxy.
- caveat: convex (like CLD baselines) uses the unlabeled target pool for feature
  normalization; plain lora/finetune don't. Stricter bar = ea_lora (run later).
- decision: KEPT — proxy beats bar. Promoting to full 9-subject sweep (convex + baselines).
- full sweep (200Hz) ran convex subj1-8 (~0.50-0.64, flat, no dip) then STOPPED to fix the
  data-input issues below before establishing official numbers.

## AUDIT (2026-05-31) — MIRepNet data input
User flagged possible MIRepNet input issues. Findings:
- channel map 22→45: CORRECT — MOABB returns the exact assumed order, no scrambling.
- sampling rate: BUG-ish. BCIC-IV-2a is native 250 Hz; repo downsampled to 200 (default
  target_sfreq, not overridden by MIREPNET cfg) then MIRepNet upsamples 200→250 — a lossy
  round-trip. Native 250 Hz zero-shot LOSO: subj1 +0.051, subj2 +0.012. → FIXED: run_local
  now builds MIRepNet data at target_sfreq=250 (cache bciciv2a_mirepnet250_cache).
- EA normalization: MIRepNet pretrained with EA, get_features uses per-channel z-score
  ("approx EA"). Tested external EA @250Hz: subj1 -0.001, subj2 -0.013 → EA does NOT help;
  keep z-score, use_ea=False. (Matches the plot's weak EA variants.)
- CAVEAT: source-FT is GPU-nondeterministic (~±0.05 single-subject zero-shot across
  "identical" runs). Mitigated for comparison by sharing the SAME disk-cached source-FT
  backbone between convex and baselines (disk cache key = backbone+split+seed+cfg).

## iter 2 — corrected input @250Hz (2026-05-31)
- change: harness now feeds MIRepNet native 250 Hz (no convex_calib code change). Re-running
  convex + sft_lora + sft_finetune proxy at 250 Hz for the official, corrected comparison.
- result (proxy, subj1-4, 250Hz, same cached source-FT backbone):
    sft_finetune 0.6014  (.596/.607/.601)
    sft_lora     0.5997  (.592/.607/.601)
    convex       0.5919  (.591/.598/.587)   <- now BEHIND by ~0.008-0.010
- decision: HONEST NEGATIVE. The 200Hz "win" did NOT survive the input fix — the better
  250Hz backbone helps direct calibration (lora/finetune) more than the source∪cal convex
  head. The combined-fit (cal_balance=1.0) likely dilutes target signal now that features
  are stronger. iter-1 method is NOT above LoRA at the correct input. Must keep iterating.
- new bar to beat (proxy): lora 0.5997 / finetune 0.6014. Next: sweep cal_balance (more
  target emphasis) + beta/n_neurons on the cached 250Hz features.
