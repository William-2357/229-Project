# Autonomous loop — per-iteration playbook

This is the prompt the `/loop` runs each iteration. It does ONE research iteration to push
the convex method above LoRA on source-pretrained MIRepNet, then stops (the loop re-fires).

## State (read every iteration)
- `research/program.md` (thesis, rules, backlog) — authoritative.
- `research/journal.md` (last ~3 entries) — what's been tried.
- `research/leaderboard.json` — current bars. **250 Hz regime is official.**
  Proxy bars to beat (subj 1-4, same source-FT MIRepNet backbone):
  **sft_lora 0.5997, sft_finetune 0.6014.** Current convex best ≈ 0.597 (tied, not ahead).

## Where convex loses (diagnosis)
Convex ties LoRA at K=0.5 (robustness zone) but loses at K=1-2 because LoRA adapts the
backbone on calibration while convex keeps it frozen + fits a head. Pure head-tuning
plateaus at the tie. To get a CLEAR win, the representation likely has to adapt too.

## Priority backlog (one per iteration; highest ROI first)
1. **K-adaptive `cal_balance`** — low at low K (robust), higher at high K (exploit more cal).
   Quick: add a schedule; sweep with `research/sweep_convex_head.py` (extend it).
2. **Source reweighting toward target** — weight/select source rows by similarity to the
   unlabeled target pool before the convex solve; sharpens the target boundary.
3. **Convex head on lightly-adapted features** (LoRA/finetune the backbone a little, then
   the convex head on top) — best-of-both; tests whether convex head > linear head on the
   SAME adapted features.
4. **CRONOS-AM alt-min** — alternate (convex head solve) ↔ (feature/last-block update) for
   2-3 rounds. The paper's core method; biggest lever, most work.
5. EA / per-class centering / ensemble — smaller.

## The iteration (do exactly this)
1. Pick the top untried idea. Form one falsifiable hypothesis.
2. Edit ONLY `adaptation/convex_calib.py` (small diff). For hyperparam searches, extend
   `research/sweep_convex_head.py` (reuses cached features — fast) to find the config, THEN
   codify the winner into HPARAMS.
3. Validate via the official harness:
   `python research/run_local.py --proxy --tag iterN_slug`
4. `python research/leaderboard_update.py --run research/runs/iterN_slug__proxy.json --iter N`
5. If proxy score **> 0.5997 (clears LoRA)** AND > current convex best → run
   `python research/run_local.py --full --tag iterN_slug` (official 9-subject curve);
   also run the lora/finetune full baselines once if not yet done, for the full comparison.
   Else if it beats current convex best but not LoRA → keep it (progress) but don't full-sweep.
   Else → `git checkout adaptation/convex_calib.py` (revert to last committed winner).
6. Append a journal entry (hypothesis / change / proxy result vs bars / decision). Be HONEST
   about negatives and noise (~±0.01 on the 4-subject proxy; don't claim wins inside noise).
7. Commit on `auto`: `git commit -am "auto: iter N — <summary> (proxy=<x> vs lora 0.600)"`.
   **NEVER add a Co-Authored-By trailer.**
8. Stop. The loop fires the next iteration.

## Guardrails
- Never edit the harness (`run_local.py`, `evaluation/*`, `data/*`, `models/*`, other
  `adaptation/*`). Only `convex_calib.py` (+ `sweep_convex_head.py` for search).
- Keep the BaseAdapter interface intact. Deterministic seed=42.
- No test leakage: fit only on source + the K calibration trials; normalize with the
  unlabeled target pool (allowed, like the CLD baselines).
- Source-FT is disk-cached + shared with baselines → fair + fast; don't change its config.
