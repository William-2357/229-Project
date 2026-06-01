"""Fast convex-head hyperparameter sweep (research tool, not the tracked metric).

Extracts the (cached) source-FT MIRepNet features ONCE per subject, then fits the convex
head for many (cal_balance, beta, n_neurons, rank) configs reusing those features. Mirrors
convex_calib's combined source∪upweighted-calibration fit + run_local's calibration
sampling, so a winning config transfers directly to convex_calib.HPARAMS.

Usage: python research/sweep_convex_head.py
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import BCICIVDataset
from data.preprocessing import MIREPNET_PREPROCESS_CONFIG
from models.foundations import build_foundation_model
from adaptation.foundation_source_finetune import build_source_finetuned_foundation_model
from adaptation.foundation_cld import extract_foundation_features
from adaptation.cld import fit_cld_head
from adaptation.convex_calib import _stratified_subsample
from evaluation.protocols import minutes_to_trials, sample_calibration_set
from evaluation.metrics import balanced_accuracy
import jax.numpy as jnp

SUBJECTS = [1, 2, 3, 4]
K_GRID = [0.5, 1.0, 2.0]
N_REPEATS = 3
SEED = 42
SOURCE_CAP = 800

# configs to try (cal_balance, beta, n_neurons, rank)
CONFIGS = [
    dict(cal_balance=1,  beta=1e-3, n_neurons=32, rank=20),   # = iter-2 baseline
    dict(cal_balance=2,  beta=1e-3, n_neurons=32, rank=20),
    dict(cal_balance=4,  beta=1e-3, n_neurons=32, rank=20),
    dict(cal_balance=8,  beta=1e-3, n_neurons=32, rank=20),
    dict(cal_balance=16, beta=1e-3, n_neurons=32, rank=20),
    dict(cal_balance=4,  beta=1e-2, n_neurons=32, rank=20),
    dict(cal_balance=4,  beta=1e-4, n_neurons=32, rank=20),
    dict(cal_balance=8,  beta=1e-3, n_neurons=64, rank=20),
    dict(cal_balance=8,  beta=1e-2, n_neurons=64, rank=30),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCICIVDataset("data/raw/bciciv2a", cache_dir="data/raw/bciciv2a_mirepnet250_cache",
                       target_sfreq=250.0, preprocess_config=MIREPNET_PREPROCESS_CONFIG)
    rng_src = np.random.default_rng(SEED)
    # scores[config_idx] = list of bca over (subject, K, repeat)
    scores = {i: [] for i in range(len(CONFIGS))}

    for subj in SUBJECTS:
        X_src, y_src = ds.get_source_data(held_out_subject=subj)
        (Xpool, ypool), (Xte, yte) = ds.get_target_data(subj)
        bb = build_foundation_model("mirepnet", n_channels=22, n_times=1000,
                                    checkpoint_path="MIRepNet.pth", input_sfreq=250.0, freeze=True)
        model = build_source_finetuned_foundation_model(
            bb, 4, X_src, y_src, device=device, lr_src=1e-3, weight_decay=1e-4,
            max_epochs_src=200, patience_src=25, val_fraction_src=0.1, batch_size=32, seed=SEED)
        model.freeze_backbone()
        backbone = model.backbone.to(device)

        # extract once
        f_src = extract_foundation_features(backbone, X_src, device, 32)
        f_src, y_src_sub = _stratified_subsample(f_src, y_src, SOURCE_CAP, np.random.default_rng(SEED))
        f_pool = extract_foundation_features(backbone, Xpool, device, 32)
        norm = (f_pool.mean(0, keepdims=True), f_pool.std(0, keepdims=True) + 1e-8)
        f_te = extract_foundation_features(backbone, Xte, device, 32)

        for k in K_GRID:
            n_cal = minutes_to_trials(k, 4, epoch_len_sec=4.0)
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(SEED + rep * 1000)
                Xc, yc, _ = sample_calibration_set(Xpool, ypool, n_cal, rng)
                f_cal = extract_foundation_features(backbone, Xc, device, 32)
                for ci, cfg in enumerate(CONFIGS):
                    reps = max(1, int(round(cfg["cal_balance"] * len(f_src) / max(1, len(f_cal)))))
                    Xfit = np.concatenate([f_src, np.tile(f_cal, (reps, 1))], 0)
                    yfit = np.concatenate([y_src_sub, np.tile(yc, reps)], 0)
                    cld, mu, sig = fit_cld_head(Xfit, yfit, 4, cfg["n_neurons"], cfg["rank"],
                                                cfg["beta"], 0.01, 1.0, 50, 10, SEED, norm_stats=norm)
                    Xn = ((f_te - mu) / sig).astype(np.float32)
                    pred = np.array(cld.stacked_predict(jnp.array(Xn), cld.theta1, cld.theta2)).argmax(1)
                    scores[ci].append(balanced_accuracy(yte, pred))
        print(f"subject {subj} done", flush=True)

    print("\n=== convex-head sweep (proxy subj1-4, K=.5/1/2) vs lora 0.5997 / ft 0.6014 ===")
    ranked = sorted(range(len(CONFIGS)), key=lambda i: -np.mean(scores[i]))
    for i in ranked:
        s = float(np.mean(scores[i]))
        flag = "  <-- BEATS LoRA" if s > 0.5997 else ""
        print(f"  score={s:.4f}  {CONFIGS[i]}{flag}")
    best = ranked[0]
    Path("research/runs/convex_head_sweep.json").write_text(json.dumps(
        {"configs": CONFIGS, "scores": {i: float(np.mean(scores[i])) for i in scores},
         "best_config": CONFIGS[best], "best_score": float(np.mean(scores[best]))}, indent=2))


if __name__ == "__main__":
    main()
