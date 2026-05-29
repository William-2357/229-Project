"""Save and load experiment results as JSON + CSV."""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime


def _make_json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    return obj


def save_result(
    result: dict,
    output_dir: str | Path,
    dataset: str,
    backbone: str,
    method: str,
    subject_id: int | None = None,
    k_minutes: float | None = None,
) -> Path:
    """Save a single result dict as JSON.

    File path: output_dir/{dataset}/{backbone}/{method}/
        subject_{id}_k{k}.json  (K-minute sweep)
        subject_{id}_loso.json  (LOSO)
        subject_{id}_within.json (within-subject CV)
    """
    out_dir = Path(output_dir) / dataset / backbone / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if subject_id is not None and k_minutes is not None:
        fname = f"subject_{subject_id:02d}_k{k_minutes:.1f}.json"
    elif subject_id is not None:
        protocol = result.get("protocol", "result")
        fname = f"subject_{subject_id:02d}_{protocol}.json"
    else:
        fname = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    result["_meta"] = {
        "dataset": dataset,
        "backbone": backbone,
        "method": method,
        "saved_at": datetime.now().isoformat(),
    }

    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(_make_json_serializable(result), f, indent=2)
    return out_path


def load_results(
    output_dir: str | Path,
    dataset: str,
    backbone: str,
    method: str,
) -> list[dict]:
    """Load all JSON result files for a given configuration."""
    result_dir = Path(output_dir) / dataset / backbone / method
    if not result_dir.exists():
        return []
    results = []
    for fpath in sorted(result_dir.glob("*.json")):
        with open(fpath) as f:
            results.append(json.load(f))
    return results


def results_to_csv(results: list[dict], out_path: str | Path) -> None:
    """Flatten a list of result dicts to CSV."""
    if not results:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat_results = [_flatten_dict(r) for r in results]
    all_keys = sorted({k for r in flat_results for k in r})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in flat_results:
            writer.writerow({k: row.get(k, "") for k in all_keys})


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compile_summary_table(
    output_dir: str | Path,
    dataset: str,
    backbone: str,
    methods: list[str],
    k_minutes_list: list[float],
) -> list[dict]:
    """Build summary table: one row per (method, k_minutes) with mean BCA."""
    rows = []
    for method in methods:
        results = load_results(output_dir, dataset, backbone, method)
        # Group by k_minutes
        by_k: dict[float, list[float]] = {}
        for r in results:
            k = r.get("k_minutes")
            bca = r.get("bca")
            if k is not None and bca is not None:
                by_k.setdefault(k, []).append(float(bca))

        for k in k_minutes_list:
            bcas = by_k.get(k, [])
            rows.append({
                "method": method,
                "k_minutes": k,
                "mean_bca": float(np.mean(bcas)) if bcas else float("nan"),
                "std_bca": float(np.std(bcas)) if bcas else float("nan"),
                "n_results": len(bcas),
            })
    return rows


def print_summary_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found.")
        return
    methods = sorted({r["method"] for r in rows})
    k_vals = sorted({r["k_minutes"] for r in rows})

    # Header
    col_w = 12
    header = f"{'Method':<20}" + "".join(f"K={k:<{col_w-2}}" for k in k_vals)
    print(header)
    print("-" * len(header))

    for method in methods:
        row_str = f"{method:<20}"
        for k in k_vals:
            match = [r for r in rows if r["method"] == method and r["k_minutes"] == k]
            if match:
                bca = match[0]["mean_bca"]
                row_str += f"{bca:.3f}{'':<{col_w - 5}}"
            else:
                row_str += f"{'N/A':<{col_w}}"
        print(row_str)
