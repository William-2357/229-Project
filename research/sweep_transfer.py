"""Fast sweep of the two-stage convex-transfer anchor (research tool, not the tracked metric).

Extracts frozen source-FT MIRepNet features ONCE per subject, solves the source convex head
v_bar once, then sweeps (transfer_stage2, anchor_a) reusing those features + anchor. Mirrors
convex_calib._transfer_head + run_local's calibration sampling, so a winning config transfers
directly to convex_calib.HPARAMS. Matches the unbiased proxy: 9 subjects, K=[1,10,30].

Sanity: ('source_cal', a=0) ≈ frozen-convex iter-3 (0.582) — validates the fixed-gate solver.

Usage: python research/sweep_transfer.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import BCICIVDataset
from data.preprocessing import MIREPNET_PREPROCESS_CONFIG
from models.foundations import build_foundation_model
from adaptation.foundation_source_finetune import build_source_finetuned_foundation_model
from adaptation.foundation_cld import extract_foundation_features
from adaptation.convex_calib import _stratified_subsample
from adaptation.convex_transfer import sample_gates, build_fixed_gate_model, anchored_admm
from evaluation.protocols import minutes_to_trials, sample_calibration_set
from evaluation.metrics import balanced_accuracy
import jax, jax.numpy as jnp

K_GRID = [1.0, 10.0, 30.0]
N_REPEATS = 3
SEED = 42
SOURCE_CAP = 800
N_NEURONS, BETA, RHO, RANK = 32, 1e-4, 0.01, 20
CAL_BALANCE = 4.0
AP = {'rank': RANK, 'beta': BETA, 'gamma_ratio': 1.0, 'admm_iters': 50, 'pcg_iters': 10, 'check_opt': False}

# (transfer_stage2, anchor_a)
CONFIGS = [
    ("cal", 0.01), ("cal", 0.1), ("cal", 1.0), ("cal", 10.0),
    ("source_cal", 0.0), ("source_cal", 0.1), ("source_cal", 1.0),
]


def solve_target(f_src, y_src, f_cal, yc, mu, sig, G, v_bar, stage2, a):
    if stage2 == "source_cal":
        reps = max(1, int(round(CAL_BALANCE * len(f_src) / max(1, len(f_cal)))))
        Xfit = np.concatenate([f_src, np.tile(f_cal, (reps, 1))], 0)
        yfit = np.concatenate([y_src, np.tile(yc, reps)], 0)
    else:
        Xfit, yfit = f_cal, yc
    Xn = ((Xfit - mu) / sig).astype(np.float32)
    tgt = build_fixed_gate_model(Xn, yfit, 4, N_NEURONS, BETA, RHO, jax.random.PRNGKey(SEED + 1), G)
    anchored_admm(tgt, AP, v_anchor=v_bar, anchor_a=a)
    return tgt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCICIVDataset("data/raw/bciciv2a", cache_dir="data/raw/bciciv2a_mirepnet250_cache",
                       target_sfreq=250.0, preprocess_config=MIREPNET_PREPROCESS_CONFIG)
    scores = {c: [] for c in CONFIGS}
    per_k = {c: {k: [] for k in K_GRID} for c in CONFIGS}

    for subj in ds.subject_ids:
        X_src, y_src = ds.get_source_data(held_out_subject=subj)
        (Xpool, ypool), (Xte, yte) = ds.get_target_data(subj)
        bb = build_foundation_model("mirepnet", n_channels=22, n_times=1000,
                                    checkpoint_path="MIRepNet.pth", input_sfreq=250.0, freeze=True)
        model = build_source_finetuned_foundation_model(
            bb, 4, X_src, y_src, device=device, lr_src=1e-3, weight_decay=1e-4,
            max_epochs_src=200, patience_src=25, val_fraction_src=0.1, batch_size=32, seed=SEED)
        model.freeze_backbone()
        backbone = model.backbone.to(device)

        f_src = extract_foundation_features(backbone, X_src, device, 32)
        f_src, y_src_sub = _stratified_subsample(f_src, y_src, SOURCE_CAP, np.random.default_rng(SEED))
        f_pool = extract_foundation_features(backbone, Xpool, device, 32)
        mu, sig = f_pool.mean(0, keepdims=True), f_pool.std(0, keepdims=True) + 1e-8
        f_te = extract_foundation_features(backbone, Xte, device, 32)
        d = f_src.shape[1]

        # Stage 1: source convex head v_bar (fixed gates G), once per subject
        key = jax.random.PRNGKey(SEED)
        G, key = sample_gates(d, N_NEURONS, key)
        src = build_fixed_gate_model(((f_src - mu) / sig).astype(np.float32), y_src_sub, 4,
                                     N_NEURONS, BETA, RHO, key, G)
        anchored_admm(src, AP, v_anchor=None, anchor_a=0.0)
        v_bar = src.v

        for k in K_GRID:
            n_cal = minutes_to_trials(k, 4, epoch_len_sec=4.0)
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(SEED + rep * 1000)
                Xc, yc, _ = sample_calibration_set(Xpool, ypool, n_cal, rng)
                f_cal = extract_foundation_features(backbone, Xc, device, 32)
                for cfg in CONFIGS:
                    tgt = solve_target(f_src, y_src_sub, f_cal, yc, mu, sig, G, v_bar, cfg[0], cfg[1])
                    Xn = ((f_te - mu) / sig).astype(np.float32)
                    pred = np.array(tgt.stacked_predict(jnp.array(Xn), tgt.theta1, tgt.theta2)).argmax(1)
                    bca = balanced_accuracy(yte, pred)
                    scores[cfg].append(bca); per_k[cfg][k].append(bca)
        print(f"subject {subj} done", flush=True)

    print("\n=== transfer-anchor sweep (9 subj, K=[1,10,30]) vs frozen-convex 0.582 / lora 0.5955 ===")
    ranked = sorted(CONFIGS, key=lambda c: -np.mean(scores[c]))
    out = {}
    for c in ranked:
        s = float(np.mean(scores[c]))
        pk = {k: round(float(np.mean(per_k[c][k])), 4) for k in K_GRID}
        flag = "  <-- beats frozen-convex" if s > 0.582 else ""
        print(f"  stage2={c[0]:<11} a={c[1]:<6} score={s:.4f}  per_k={pk}{flag}")
        out[f"{c[0]}_a{c[1]}"] = {"score": s, "per_k": pk}
    Path("research/runs/transfer_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
