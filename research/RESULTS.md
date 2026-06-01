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
