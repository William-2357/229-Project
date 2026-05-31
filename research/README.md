# Convex EEG Calibration — Autoresearch Harness

A Karpathy-style ([karpathy/autoresearch](https://github.com/karpathy/autoresearch))
autonomous research loop. The agent iterates on **one** file to push test BCA on the
BCIC-IV-2a calibration benchmark using convex / convex-reformulated NN methods.

## Layout

| Path | Role | Edited by loop? |
|---|---|---|
| `program.md` | Standing brief: thesis, metric, rules, idea backlog | no (append backlog only) |
| `adaptation/convex_calib.py` | **The one editable file** — the convex method | **YES** |
| `run_local.py` | Fixed eval harness → single tracked metric | no |
| `leaderboard.json` | Best proxy + full scores, history | bookkeeping only |
| `leaderboard_update.py` | Atomic leaderboard update from a run json | no |
| `journal.md` | Append-only iteration log | append only |
| `prep_data_moabb.py` | One-time: fetch BCIC-IV-2a via MOABB → repo npz | no |
| `runs/*.json` | Per-iteration run results | written by harness |

## Metric

`score` = mean test BCA over all (subject, K) cells. `low_k_score` = mean over the two
smallest K (the low-resource regime where convexity should win). Higher is better.

## One iteration (the loop)

1. Read `program.md` + recent `journal.md` + `leaderboard.json`.
2. Form one falsifiable hypothesis (convexity thesis / backlog).
3. Edit `adaptation/convex_calib.py`.
4. Proxy: `python research/run_local.py --proxy --tag <slug>` (4 subjects, low-K).
5. `python research/leaderboard_update.py --run research/runs/<slug>__proxy.json --iter <N>`.
   - beats proxy best → full: `python research/run_local.py --full --tag <slug>` then
     `leaderboard_update.py` on the `__full.json`.
   - else → revert `convex_calib.py` to the last winning version.
6. Append a journal entry. Commit on `auto`:
   `auto: iter <N> — <summary> (proxy=<x> full=<y/-->)`. **No `Co-Authored-By` trailer.**

## Setup (already done)

```bash
# env: torch cu128 (Blackwell), jax[cuda12], jaxcld, moabb, EEG stack -> .venv
python research/prep_data_moabb.py            # BCIC-IV-2a -> data/raw/bciciv2a/*.npz
python research/run_local.py --proxy --tag t  # smoke test (auto-uses real data if cached)
```

Backbone for the convex head is a frozen source-trained EEGNet (foundation checkpoints
are not present locally). Single RTX 5090, CUDA.
