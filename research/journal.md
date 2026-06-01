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

## iter 3 — cal_balance=4, beta=1e-4 (2026-06-01)
- hypothesis: the stronger 250Hz backbone needs more target emphasis in the convex head
  fit; lighter group-lasso. (From sweep_convex_head.py over cal_balance/beta/n_neurons.)
- change: convex_calib HPARAMS cal_balance 1->4, beta 1e-3->1e-4. (sweep best of 9 configs;
  cal_balance 8-16 and beta 1e-2 are worse.)
- proxy (subj1-4, 250Hz): score=0.5971 low_k=0.6036 per_k={.5:.610, 1:.597, 2:.584}.
- vs bars: **WINS low-K** — K=0.5 0.610 vs lora .592 / ft .596 (+0.014-0.018); low_k 0.6036
  vs lora .599 / ft .601. Loses at K=2 (.584 vs .601). Overall 0.5971 ~tied with lora .5997.
- decision: KEPT (new convex best, +0.005 over iter-2; clear low-K leadership = the thesis).
  Not full-swept yet (doesn't clear lora OVERALL). Next idea: K-adaptive cal_balance to
  recover K>=2 (low cal_balance at low K, higher at high K) → win across the whole curve.

## → AUTONOMOUS LOOP handed off here (see research/LOOP.md)
Machinery validated end-to-end (reconstruct infra, source-FT MIRepNet, fair baselines,
sweep, proxy/full eval, honest journaling, no-coauthor commits). Loop pursues the backlog
to turn the low-K lead into a whole-curve win over LoRA.

## iter 4 — K-adaptive cal_balance (2026-06-01)  [REVERTED]
- hypothesis: raise cal_balance at higher K (more, more-reliable cal trials) to recover the
  K>=2 regime where lora leads.
- change: cal_balance_mode=adaptive, cb=clip(n_cal/4, 4, 12). K0.5/1→cb=4, K2→cb=7.5.
- proxy (subj1-4, 250Hz): score=0.5928 low_k=0.6036 per_k={.5:.610, 1:.597, 2:.571}.
  K0.5/1 unchanged (cb still 4); K2 got WORSE (.571 vs iter-3 .584).
- decision: REVERTED (0.5928 < iter-3 0.5971). Raising cal_balance at high K HURTS
  (matches sweep: cb=8+ worse). Optimal cal_balance ≈ constant 2-4 across K → head/source-cal
  rebalancing is EXHAUSTED. The K>=2 gap to lora needs REPRESENTATION adaptation, not head
  reweighting. Next (iter-5): convex head on calibration-adapted features — let the backbone
  adapt on calibration (like finetune), then put the convex ReLU head on top instead of a
  linear head. Tests whether the convex head beats a linear head on the SAME adapted features.

## iter 5 — smart-hybrid backbone adaptation (2026-06-01)  [REVERTED]
- hypothesis: fine-tune backbone on calibration when n_cal>=20 (K>=2), else frozen; then
  convex head on source∪cal adapted features. Adapt the representation only where safe.
- change: adapt_backbone=True, adapt_min_cal=20, _adapt_on_cal (lr_tgt=1e-5, 100ep, patience15).
- proxy (subj1-4, 250Hz): score=0.5965 low_k=0.6036 per_k={.5:.610, 1:.597, 2:.582}.
  K0.5/1 unchanged (frozen, n_cal<20); K2 (adapted) .582 ≈ iter-3 frozen .584 — NO GAIN, and
  below lora K2 .601. (subj1 helped in smoke, didn't hold across subjects.)
- decision: REVERTED (0.5965 < iter-3 0.5971). Adapting backbone then convex-on-source∪cal is
  a wash — the cal-shifted source features likely DILUTE the fit (also why finetune+convex .582
  < finetune+linear .601 at K2). Next (iter-6): when backbone IS adapted, fit convex head on
  CALIBRATION ONLY (drop source; the adapted backbone already encodes target structure).

## iter 6 — adapt backbone + convex on CALIBRATION ONLY (2026-06-01)  [REVERTED]
- hypothesis: with the backbone adapted to target, fit convex head on cal-only (source no
  longer needed and dilutes); should unlock the adapted representation at K>=2.
- change: adapt_backbone + (when adapted) X_fit = calibration features only.
- proxy (subj1-4, 250Hz): score=0.5701 low_k=0.6036 per_k={.5:.610, 1:.597, 2:.503}.
  K=2 CRATERED to .503 (vs iter-3 .584, lora .601).
- decision: REVERTED (0.5701 << iter-3 0.5971). Convex 2-layer ReLU head on ~30 points is
  badly underdetermined (the SAME dip mechanism). The source∪cal anchor is NECESSARY for the
  convex head's well-posedness — but source dilutes once the backbone is target-adapted.
  => representation-adaptation line EXHAUSTED (iter-5 dilutes, iter-6 underdetermines).
- CONCLUSION: best convex = iter-3 (frozen backbone + convex on source∪cal, cal_balance=4,
  beta=1e-4). Consolidating: run the official full 9-subject sweep (convex iter-3 + sft_lora
  + sft_finetune at 250Hz) to resolve the proxy tie at scale and produce the definitive curve.
  Remaining stretch idea if a whole-curve win is still wanted: CRONOS-AM alt-min (convex
  representation adaptation), which neither dilutes nor underdetermines like iters 5-6 did.

## OFFICIAL FULL 9-SUBJECT RESULT (2026-06-01) — convex (iter-3) vs sft_lora, 250Hz
Same source-FT MIRepNet backbone for both. K=0.5..30, n_repeats=5.
    K     convex   lora     Δ
    0.5   0.550    0.559    lora +0.009
    1.0   0.557    0.565    lora +0.007
    2.0   0.556    0.566    lora +0.010
    5.0   0.584    0.573    convex +0.011   (convex's only win)
    10    0.587    0.587    tie
    15    0.597    0.623    lora +0.026
    30    0.602    0.634    lora +0.033
    mean  0.576    0.587    lora +0.011

**HONEST BOTTOM LINE — the premise is NOT supported on the full benchmark.** LoRA beats
convex at every K except K=5, and overall (0.587 vs 0.576). The earlier "convex wins low-K"
was a PROXY ARTIFACT: the 4-subject proxy is dominated by subject 1 (convex 0.705 vs lora
0.650 @K=0.5), who is 1/4 of the proxy but 1/9 of the full set. Across all 9 subjects convex's
low-K edge vanishes. LoRA's backbone adaptation wins decisively at high K (15-30), which the
frozen-feature convex head cannot match.

METHODOLOGICAL FINDING: the subj-1-4 proxy systematically OVERSTATES convex → unreliable for
loop selection. Any further iterations must validate on all 9 subjects (or a balanced subset).

STATUS: iters 1-6 explored head tuning + representation adaptation; none beat LoRA on the full
benchmark. Convex is COMPETITIVE (within ~0.01 at low/mid K, wins K=5, wins subject 1 outright)
but not superior. Remaining principled lever = CRONOS-AM alt-min (uncertain, ~1h/iter to
validate on full). PAUSING the autonomous loop here to report this result-overturning finding
and get a decision: invest in CRONOS-AM vs. accept the honest result.
[User chose: implement CRONOS-AM, fix the biased proxy first.]

## PROXY FIX (2026-06-01)
run_local proxy was subj1-4 / K=[.5,1,2] (biased toward convex via subject 1). Fixed to ALL
9 subjects (source-FT disk-cached so affordable), K=[1,10,30] spanning low/mid/high. New
bars (9-subj, same backbone): lora 0.5955, convex iter-3 0.582.

## iter 7 — CRONOS-AM alternating minimization (2026-06-01)  [REVERTED]
- hypothesis: adapt the representation CONVEXLY via a light low-rank feature-space adapter A
  trained THROUGH the fixed convex head (alternate: solve head on A(source∪cal) <-> grad-update
  A on cal CE), gated to K>=5 (adapter_min_cal=60; at K=1 the adapter overfits). Avoids iter-5
  dilution / iter-6 underdetermination, and is cheap (256-d feature space, no backbone fwd).
- change: altmin_rounds=2, rank=16, 60 steps; _head_forward (differentiable relu(Z@W1)@W2),
  _solve_head / _update_adapter.
- proxy (9 subj, K=[1,10,30]): score=0.5816 per_k={1:.555, 10:.585, 30:.6045}.
  vs iter-3 {1:.557,10:.587,30:.602}=0.582 -> NO IMPROVEMENT (within noise). vs lora 0.5955.
- decision: REVERTED (0.5816 ≈ iter-3 0.582). The adapter barely moves the representation —
  re-solving the global convex head absorbs its change, and the frozen MIRepNet features cap
  what any head/feature-adapter can do. K=30 stays .605 (lora .634).

## ===== FINAL HONEST CONCLUSION (2026-06-01) =====
Across 7 iterations on the corrected, fair, full benchmark (source-pretrained MIRepNet @250Hz,
same backbone for all methods): the convex calibration head is COMPETITIVE but does NOT beat
LoRA. Best convex = iter-3 (frozen backbone + convex ReLU head on source∪upweighted-cal,
cal_balance=4, beta=1e-4): full-9 mean BCA 0.576 vs lora 0.587 (-0.011); wins only K=5; ties
mid-K; trails at K=15-30. 
Tried and failed to beat LoRA: head/source-cal rebalancing (iter3-4), heavy-backbone adaptation
(iter5 dilutes, iter6 underdetermines), CRONOS-AM light feature-adapter (iter7, no gain).
ROOT CAUSE: LoRA/finetune ADAPT THE REPRESENTATION (backbone); the convex method keeps the
backbone frozen and adapts only a head/feature-map, which cannot match backbone adaptation at
high K. The thesis (convex robust at low resource) holds DIRECTIONALLY — convex is closest to
LoRA at low K and wins on the easiest subject (subj1: .705 vs .650 @K=0.5) with provable global
optimality — but it is not superior overall on this strong FM backbone.
=> Consolidating and reporting. Autonomous loop stopped here (convex design space explored).

## iter 8 — LoRA + convex head (HYBRID, user-requested) (2026-06-01)  [KEPT — BEATS LoRA]
- hypothesis: the frozen-feature limit is the problem. Use sft_lora's exact LoRA adaptation
  (rank 8) to adapt the representation on calibration, then replace lora's LINEAR head with
  the convex ReLU head (on source∪upweighted-cal of the LoRA-adapted features). Tests: does
  the convex head beat a linear head on the SAME adapted representation?
- change: use_lora=True; _lora_adapt (peft LoRA, finetune on cal, merge_and_unload), then
  iter-3 convex head on the merged backbone's features.
- proxy (9 subj, K=[1,10,30]): score=0.6110 per_k={1:.558, 10:.6305, 30:.645}.
  vs sft_lora 0.5955: +0.0155 OVERALL; K10 .6305 vs .587 (+0.044!), K30 .645 vs .634 (+0.011),
  K1 .558 vs .565 (~tie). vs convex iter-3 0.582: +0.029.
- decision: KEPT — NEW BEST and BEATS LoRA on the proxy (margin >> ~0.01 noise, esp. K10).
  The convex head adds real value on top of LoRA's representation. Promoting to FULL 9-subject
  sweep (all 7 K, n_repeats=5) to confirm the whole-curve win. NOTE: this is LoRA+convex
  (representation adaptation via LoRA + convex classifier), the user-requested hybrid.
- FULL 9-subject sweep (all 7 K, n_repeats=5) — CONFIRMS the win:
    K       0.5    1     2     5     10    15    30    mean
    L+cvx  .557  .559  .557  .580  .625  .647  .642  0.5952
    lora   .559  .565  .566  .573  .587  .623  .634  0.5868
    Δ      -.002 -.006 -.009 +.007 +.038 +.024 +.008 +0.0084
  LoRA+convex BEATS lora overall (0.595 vs 0.587). Ties at low K (≤2), WINS decisively at
  K=10-15 (+0.038/+0.024, >> noise). Final full ranking:
    **LoRA+convex 0.595 > sft_lora 0.587 > convex-frozen 0.576 > sft_finetune 0.572.**
  => The convex head beats a LINEAR head on the SAME LoRA-adapted representation. The thesis
  pays off once the representation is adapted: the globally-optimal convex classifier extracts
  more from the LoRA-adapted features than lora's own linear head, especially at mid-high K.
  This is the headline positive result. Next: tune the hybrid (lora_rank, cal_balance/beta on
  adapted features) to widen the margin, esp. recover the small low-K gap.

## iter 9 — LoRA+convex with lora_rank=16 (2026-06-01)  [REVERTED]
- hypothesis: more LoRA capacity (rank 8->16) widens the mid/high-K margin.
- proxy (9 subj, K=[1,10,30]): score=0.6048 per_k={1:.555, 10:.6183, 30:.6411}.
  vs iter-8 (rank 8) proxy 0.6110 {1:.558,10:.6305,30:.645} -> WORSE (K10 .618 vs .631).
- decision: REVERTED. rank 16 overfits the calibration; rank 8 is optimal. iter-8 (LoRA+convex,
  rank 8) stands as the FINAL WINNER. Hybrid is well-characterized -> consolidating.

## ===== FINAL RESULT (2026-06-01) =====
WIN: **LoRA+convex (iter-8)** beats LoRA on the fair full 9-subject benchmark (250Hz, same
source-FT MIRepNet): mean BCA 0.595 vs sft_lora 0.587 vs convex-frozen 0.576 vs sft_finetune
0.572. Recipe: source-FT MIRepNet -> LoRA-adapt (rank 8) on calibration -> convex ReLU head
(jaxcld, source∪upweighted-cal, cal_balance=4, beta=1e-4) replacing lora's linear head.
The convex head beats a linear head on the SAME LoRA-adapted representation, decisively at
K=10-15 (+0.038/+0.024); ties at low K. Validates the thesis ONCE the frozen-feature ceiling
is removed by representation adaptation. Explored 9 iterations; tuning (rank, cal_balance,
K-schedule, backbone-adapt, CRONOS-AM) is characterized — rank 8 hybrid is the optimum found.
Possible future gain: re-tune the convex-head hparams specifically on LoRA-adapted features.

## iter 10 — re-tune convex head ON LoRA-adapted features (2026-06-01)  [no gain]
- hypothesis: iter-8 inherited convex-head hparams (cal_balance=4, beta=1e-4) from the frozen
  regime; the LoRA-adapted feature distribution differs, so re-tuning may widen the margin.
- method: sweep_lora_convex.py — LoRA-adapt once per (subj,K,rep), then 6 convex configs on the
  cached adapted features (9 subj, K=[1,10,30]).
- result (ranking, controlled — same adapted features per cell):
    cal_balance=4 beta=1e-4 n=32  0.6031  <- BEST (= iter-8 config)
    cal_balance=2 beta=1e-4       0.6024
    cal_balance=4 beta=1e-3       0.6023
    cal_balance=4 beta=1e-4 n=48  0.6008
    cal_balance=8 *               0.596x  (worse)
- decision: NO GAIN — iter-8's config (cal_balance=4, beta=1e-4, n=32) is already optimal on
  LoRA-adapted features too. No code change. (Abs 0.603 vs iter-8 run_local 0.611 = RNG/path
  noise in the sweep's LoRA finetune; ranking is what matters.)

## ===== LOOP CONCLUDED (2026-06-01) — final, fully characterized =====
10 iterations. FINAL WINNER: **iter-8 LoRA+convex** (rank 8, cal_balance=4, beta=1e-4) —
full-9 0.595 vs sft_lora 0.587 vs convex-frozen 0.576 vs sft_finetune 0.572. Optimum confirmed:
lora_rank 8 best (iter-9), convex-head hparams optimal (iter-10). convex_calib.py is at this
winner. No further promising convex/hybrid lever remains within this design space. Deliverables:
research/RESULTS.md, research/kmin_results.png, research/journal.md, research/leaderboard.json.

## iters 11-13 — cross-subject-generality convex objectives (deep-research-inspired) [all negative]
User asked for DIFFERENT convex formulations maximizing cross-subject generality ("learn
general rule -> adapt"; no subject IDs). Deep-research workflow (105 agents) ranked: R2D2 ridge
meta, MetaOptNet SVM, Group-DRO, IRMv1. Implemented 3 as `generality_mode` in convex_calib
(frozen backbone, use_lora=false, isolating the objective; per-subject source via source_cache):
- iter-11 meta_r2d2 (R2D2): meta-train low-rank adapter via leave-one-source-subject-out
  episodes + closed-form differentiable ridge inner. proxy(9subj,K=[1,10,30])=0.5766
  {1:.555,10:.569,30:.605}.
- iter-12 irm (IRMv1 gradient penalty): 0.5667 {1:.536,10:.569,30:.595}.
- iter-13 group_dro (worst-subject exp-grad reweighting): 0.5662 {1:.532,10:.580,30:.587}.
- bars: frozen-convex 0.582, LoRA+convex(iter-8) 0.611, lora 0.5955.
- decision: ALL NEGATIVE. None beats frozen-convex (0.582); all far below LoRA+convex (0.611).
  meta_r2d2 ≈ frozen (tie, edges it at K=30); irm/dro lost low-K discriminability.
- INTERPRETATION (honest): a strong source-pretrained foundation backbone ALREADY encodes
  cross-subject-general features, so explicit generality objectives (meta/invariance/worst-
  subject) on FROZEN features add nothing or hurt. The performance lever here is PER-TARGET
  representation adaptation (LoRA on the calibration trials), not a better cross-subject-general
  fixed rule. Matches the research's BOIL/ANIL caveat (cross-domain needs representation change)
  and explains why LoRA+convex wins. Convex_calib default stays iter-8 (use_lora=True,
  generality_mode="none"). generality_mode code retained as a research artifact.
- FINAL (unchanged): LoRA+convex (iter-8) remains the winner — full-9 0.595 > lora 0.587.

## SPECIALIST PORT — eegnet quick test (2026-06-01)
Ported convex methods to specialists (adaptation/convex_calib_specialist.py): source-train
eegnet -> {convex: frozen + convex head on source∪cal | ft_convex: gentle full-FT on cal +
convex head | ft_linear: full-FT + linear head = finetune}. (peft-LoRA can't wrap braindecode
MaxNorm convs under torch 2.11 -> representation adaptation uses full-FT.)
Clean 3-way, SAME source-trained backbone, 9 subj, K=[1,10,30], n_repeats=2:
    ft_linear (finetune)  0.5405  {1:.529, 10:.536, 30:.557}
    ft_convex (ours)      0.5371  {1:.497, 10:.545, 30:.570}
    convex    (ours)      0.5263  {1:.490, 10:.533, 30:.556}
FINDING: on eegnet the convex head does NOT beat the linear head — ft_convex ≈ ft_linear
(within noise), finetune ahead at low K, convex ahead only at K=30. The MIRepNet convex win
(+0.038 @K=10) does NOT replicate -> the convex-head advantage is BACKBONE-DEPENDENT: strong on
rich foundation features (MIRepNet 256-d), neutral/mixed on smaller specialist features (eegnet).
Caveat: eegnet source-train is GPU-nondeterministic (~±0.02 across runs); within-run comparison
(shared backbone) is controlled, margins are within noise. Next option: shallowconv/conformer
(conformer's transformer features may behave more like MIRepNet).
