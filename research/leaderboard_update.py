"""Update research/leaderboard.json from a run result. Part of the fixed harness.

Compares a run's `score` against the current best for its mode (proxy/full) and
updates the leaderboard if it's a new best. Always appends to history. Prints whether
it was a new best so the loop can branch.

Usage:
    python research/leaderboard_update.py --run research/runs/<tag>__proxy.json --iter 3
    python research/leaderboard_update.py --run research/runs/<tag>__full.json  --iter 3
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LB = REPO / "research" / "leaderboard.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text())
    lb = json.loads(LB.read_text())
    mode = run["mode"]
    key = f"{mode}_best"
    cur = lb.get(key, {}) or {}
    cur_score = cur.get("score")
    new_score = run["score"]

    is_best = cur_score is None or new_score > cur_score
    entry = {
        "tag": run["tag"], "score": new_score, "low_k_score": run["low_k_score"],
        "iter": args.iter, "dataset": run["dataset"],
    }
    if mode == "full":
        entry.update({"per_k": run["per_k"], "k_star": run["k_star"]})

    if is_best:
        lb[key] = entry

    lb.setdefault("history", []).append({
        "iter": args.iter, "mode": mode, "tag": run["tag"],
        "score": new_score, "low_k_score": run["low_k_score"],
        "dataset": run["dataset"], "new_best": is_best,
    })
    LB.write_text(json.dumps(lb, indent=2))

    prev = "None" if cur_score is None else f"{cur_score:.4f}"
    print(f"[{mode}] iter {args.iter} tag={run['tag']} score={new_score:.4f} "
          f"(prev best {prev}) -> {'NEW BEST' if is_best else 'no improvement'}")


if __name__ == "__main__":
    main()
