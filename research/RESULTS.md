# Convex calibration on source-pretrained MIRepNet — results

Autoresearch loop (research/program.md, research/journal.md) testing the thesis that convex
NN heads are superior for low-resource EEG-BCI calibration, on BCIC-IV-2a with a
source-fine-tuned MIRepNet foundation backbone.

## Setup (fair by construction)
- Backbone: MIRepNet (HF `starself/MIRepNet`), **source-fine-tuned then frozen** — the SAME
  disk-cached backbone for every method, so only the adaptation differs.
- Data: BCIC-IV-2a via MOABB, MIRepNet preprocessing, **native 250 Hz** (fixed a lossy
  250→200→250 round-trip that cost ~0.03–0.05 BCA).
- Protocol: LOSO source; K-minute calibration sweep; mean test BCA over 9 subjects,
  n_repeats=5; K = 0.5…30 min.

## Official full 9-subject result (250 Hz)
| K (min) | **LoRA+convex** (iter-8) | sft_lora | convex-frozen (iter-3) | sft_finetune |
|--------:|-------------------------:|---------:|-----------------------:|-------------:|
| 0.5     | 0.557                    | 0.559    | 0.550 | 0.564 |
| 1       | 0.559                    | 0.565    | 0.557 | 0.564 |
| 2       | 0.557                    | 0.566    | 0.556 | 0.565 |
| 5       | 0.580                    | 0.573    | 0.584 | 0.565 |
| 10      | **0.625**                | 0.587    | 0.587 | 0.578 |
| 15      | **0.647**                | 0.623    | 0.597 | 0.584 |
| 30      | **0.642**                | 0.634    | 0.602 | 0.586 |
| **mean**| **0.595**                | 0.587    | 0.576 | 0.572 |

**Ranking: LoRA+convex 0.595 > sft_lora 0.587 > convex-frozen 0.576 > sft_finetune 0.572.**
The **LoRA+convex hybrid wins** — ties LoRA at low K (≤2), beats it decisively at K=10–15
(+0.038/+0.024). The convex head extracts more from a LoRA-adapted representation than LoRA's
own linear head. Plain (frozen-backbone) convex beats full finetune but trails LoRA. Graph:
`kmin_results.png`.

## Best convex method (iter-3)
Frozen source-FT MIRepNet → convex two-layer ReLU head (jaxcld CVX_ReLU_MLP + ADMM) fit on the
**union of source ∪ upweighted-calibration** features (`cal_balance=4`, `beta=1e-4`). One global
convex solve, well-posed at every K — this fixed the stock `sft_cld` low-K dip.

## What was tried (7 iterations)
1–3  head / source-cal rebalancing → best = iter-3 (cal_balance=4, beta=1e-4).
4    K-adaptive cal_balance → worse (optimal cal_balance ≈ constant).
5    adapt backbone + convex on source∪cal → wash (cal-shifted source dilutes the fit).
6    adapt backbone + convex on cal-only → K=2 crater (convex head underdetermined on ~30 pts).
7    CRONOS-AM (light feature-space adapter trained through the fixed convex head) → no gain.
8    **LoRA+convex hybrid → WINS**: LoRA adapts the representation, convex head replaces lora's
     linear head → 0.595, beats lora 0.587 (decisive at K=10-15).

## Honest conclusion
The frozen-backbone convex head is competitive (beats full finetune, 2nd of 3) but cannot beat
LoRA alone — because LoRA adapts the *representation* (backbone) and a head on *frozen* features
can't match that at high K. **The fix is to combine them: LoRA+convex (iter-8) beats LoRA**
(0.595 vs 0.587), since the globally-optimal convex classifier extracts more from the
LoRA-adapted features than a linear head — clearest at mid-high K (K=10-15). At low K (≤2) all
methods are within ~0.01. So the convex contribution is real and additive *on top of*
representation adaptation, validating the thesis once the frozen-feature ceiling is removed.

(Earlier frozen-only conclusion, for the record:)
On this strong FM backbone, the convex calibration head is **competitive (2nd of 3, beats full
finetune) but does not beat LoRA**. Root cause: LoRA/finetune adapt the *representation*
(backbone); a convex head/feature-adapter on *frozen* features cannot match that at high K.
The thesis holds **directionally** — convex is closest to LoRA at low K, wins at K=5, wins the
easiest subject outright (subj 1: 0.705 vs 0.650 @ K=0.5), and brings provable global
optimality / stability — but it is not uniformly superior here.

Methodological note: a 4-subject proxy (subjects 1–4) badly OVERSTATED convex (subject 1 favors
it); all reported numbers use the full 9-subject sweep, and the loop's proxy was fixed to all 9.

## Repro
```
python research/prep_data_moabb.py                                  # data
python research/run_local.py --full --tag convex --method convex_calib
python research/run_local.py --full --tag lora   --method foundation_sft_lora
python research/run_local.py --full --tag ft     --method foundation_sft_finetune
python research/plot_official.py
```

---

## Convex-in-pretraining arc (relaxed harness, iters 14–18)

After the head-only loop, the harness was relaxed to allow **convexity in the pre-training
stage**, guided by the team-paper reference (arXiv 2605.23235) on the two-stage convex transfer.
Governing principle: in a convex model "pretrain→finetune via initialization" is meaningless
(the solution is set by data+regularizer+dictionary, not the optimization path), so source
knowledge can transfer only through the **gate dictionary** or the **regularizer (anchor)**.

Built `adaptation/convex_transfer.py`: an **anchored ADMM** (v-update blended toward a
source-pretrained convex head `v_bar`; verified identical to stock jaxcld ADMM at `a=0` to 3e-8),
generalized to a per-pattern (Mahalanobis-spirit) anchor whose strength `a_i ∝ 1/Var_s(v_i)` from
a multi-task per-source-subject solve.

**Result 1 — the convex-transfer anchor is a thorough NEGATIVE.** Tested isotropic + adaptive,
cal-only + source-pooled, on frozen + LoRA features; none beats its bar (frozen-convex 0.582 /
LoRA+convex 0.611). **Key finding:** when source *data* is available offline, **pooling raw source
features into the convex fit dominates anchoring to a source-head summary.** The anchor earns its
keep in the **online/streaming** regime the paper originally targeted (source can't be re-accessed),
not this offline benchmark.

**Result 2 — a convex-head ENSEMBLE is marginal.** Averaging M independent LoRA+convex members
(each its own LoRA seed + gates). Controlled test (same members per cell): a real **+0.013**
(M=1→M=3, variance reduction, saturating at M=3, largest at high K). On the official full grid,
**ensemble M=3 = 0.598 vs iter-8 0.595 (+0.003, within noise)** — gains at K=5/30 offset by small
losses at K=0.5/10/15. Highest recorded, but not a decisive win, and 3× the compute.

**Result 3 — coupling the representation to the convex head (CRONOS-AM) is REFUTED.** iter-8 is
*decoupled*: LoRA trains via a throwaway linear head, the convex head is fit post-hoc. We coupled
them — alternate `global convex-head solve` ↔ `LoRA update on cal CE backpropped through the fixed
convex head` (the real CRONOS-AM, enabled by the relaxed harness). Both re-solve frequencies
(rounds=3 and rounds=6) give **0.589 vs iter-8 0.611 (−0.022), worse at every K.** **Why decoupling
wins (the insight):** representation learning wants a *smooth, co-adapting* objective — iter-8's
linear head rotates with the features each SGD step, a clean gradient. The convex head's power is as
a *global-optimal, post-hoc* classifier of nonlinear structure in already-good features; used as the
representation's *training target* it is static and ill-conditioned (dead ReLUs, two-sided neurons
with negative W₂) and **degrades** the features. The decoupling is **load-bearing**, not incidental.

**Standing winner unchanged: iter-8 LoRA+convex (full 0.595).** The robust lever remains
per-target representation adaptation (LoRA) + a convex head; convex transfer, ensembling, and
coupling all fail to decisively exceed it on this strong FM backbone. `convex_calib.py` default
stays `n_ensemble=1, transfer_mode=none, couple_mode=none`; `n_ensemble=3` is the highest-accuracy
variant when 3× compute is acceptable.

Repro: `CONVEX_HP='{"n_ensemble":3}' python research/run_local.py --full --tag ens3`;
`CONVEX_HP='{"couple_mode":"cronos_am"}' ... --tag coupled`; sweeps `research/sweep_transfer.py`,
`research/sweep_lora_transfer.py`, `research/sweep_ensemble.py`.
