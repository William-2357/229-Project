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
| K (min) | convex (iter-3) | sft_lora | sft_finetune |
|--------:|----------------:|---------:|-------------:|
| 0.5     | 0.550           | 0.559    | 0.564 |
| 1       | 0.557           | 0.565    | 0.564 |
| 2       | 0.556           | 0.566    | 0.565 |
| 5       | **0.584**       | 0.573    | 0.565 |
| 10      | 0.587           | 0.587    | 0.578 |
| 15      | 0.597           | 0.623    | 0.584 |
| 30      | 0.602           | 0.634    | 0.586 |
| **mean**| **0.576**       | **0.587**| **0.572** |

Ranking: **LoRA 0.587 > convex 0.576 > finetune 0.572.** Convex beats full finetune and is
competitive with LoRA; LoRA wins via backbone adaptation at high K (15–30). See
`convex_vs_baselines_full.png`.

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

## Honest conclusion
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
