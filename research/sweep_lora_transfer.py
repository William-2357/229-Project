"""iter-15: does the two-stage convex-transfer ANCHOR help ON TOP of LoRA? (research tool)

LoRA-adapts ONCE per (subject,K,repeat) — the expensive step — then on those cached adapted
features: (1) solves a source convex head v_bar with fixed gates, (2) sweeps (transfer_stage2,
anchor_a) anchored target solves. Control: ('source_cal', a=0) == iter-8 LoRA+convex (0.611).
Tests the reference's anchor channel where representation adaptation (LoRA) is already applied.

Usage: python research/sweep_lora_transfer.py
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.datasets import BCICIVDataset
from data.preprocessing import MIREPNET_PREPROCESS_CONFIG
from models.foundations import build_foundation_model, FoundationWithHead
from adaptation.foundation_source_finetune import build_source_finetuned_foundation_model
from adaptation.foundation_lora import _get_lora_target_modules
from adaptation.foundation_cld import extract_foundation_features
from adaptation.convex_calib import _stratified_subsample
from adaptation.convex_transfer import sample_gates, build_fixed_gate_model, anchored_admm
from adaptation.base import train_epoch, evaluate_model
from evaluation.protocols import minutes_to_trials, sample_calibration_set
from evaluation.metrics import balanced_accuracy
from peft import LoraConfig, get_peft_model
import jax, jax.numpy as jnp

SUBJECTS = list(range(1, 10))
K_GRID = [1.0, 10.0, 30.0]
N_REPEATS = 3
SEED = 42
SOURCE_CAP = 800
LORA_RANK = 8
N_NEURONS, BETA, RHO, RANK, CAL_BALANCE = 32, 1e-4, 0.01, 20, 4.0
AP = {'rank': RANK, 'beta': BETA, 'gamma_ratio': 1.0, 'admm_iters': 50, 'pcg_iters': 10, 'check_opt': False}

# (transfer_stage2, anchor_a)
CONFIGS = [
    ("source_cal", 0.0),   # == iter-8 LoRA+convex control (~0.611)
    ("source_cal", 0.1),
    ("source_cal", 1.0),
    ("cal", 1.0),
    ("cal", 3.0),
]


def lora_adapt(model, X_cal, y_cal, device):
    targets = _get_lora_target_modules(model, rank=LORA_RANK)
    lm = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK * 2,
                        target_modules=targets, lora_dropout=0.1, bias="none"))
    n_val = max(1, int(len(X_cal) * 0.1)); idx = np.random.permutation(len(X_cal))
    vi, ti = idx[:n_val], idx[n_val:] if len(idx) > n_val else idx
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    best, bs, pat = -1.0, None, 0
    for _ in range(100):
        train_epoch(lm, X_cal[ti], y_cal[ti], opt, device, 32)
        acc = evaluate_model(lm, X_cal[vi], y_cal[vi], device)
        if acc > best: best, bs, pat = acc, copy.deepcopy(lm.state_dict()), 0
        else:
            pat += 1
            if pat >= 15: break
    if bs is not None: lm.load_state_dict(bs)
    return lm.merge_and_unload()


def solve_target(f_src, y_src, f_cal, yc, mu, sig, G, v_bar, stage2, a):
    if stage2 == "source_cal":
        reps = max(1, int(round(CAL_BALANCE * len(f_src) / max(1, len(f_cal)))))
        Xfit = np.concatenate([f_src, np.tile(f_cal, (reps, 1))], 0)
        yfit = np.concatenate([y_src, np.tile(yc, reps)], 0)
    else:
        Xfit, yfit = f_cal, yc
    Xn = ((Xfit - mu) / sig).astype(np.float32)
    tgt = build_fixed_gate_model(Xn, yfit, 4, N_NEURONS, BETA, RHO, jax.random.PRNGKey(SEED + 1), G)
    anchored_admm(tgt, AP, v_anchor=(None if a == 0.0 else v_bar), anchor_a=a)
    return tgt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCICIVDataset("data/raw/bciciv2a", cache_dir="data/raw/bciciv2a_mirepnet250_cache",
                       target_sfreq=250.0, preprocess_config=MIREPNET_PREPROCESS_CONFIG)
    scores = {c: [] for c in CONFIGS}
    per_k = {c: {k: [] for k in K_GRID} for c in CONFIGS}
    for subj in SUBJECTS:
        X_src, y_src = ds.get_source_data(held_out_subject=subj)
        (Xpool, ypool), (Xte, yte) = ds.get_target_data(subj)
        bb = build_foundation_model("mirepnet", n_channels=22, n_times=1000,
                                    checkpoint_path="MIRepNet.pth", input_sfreq=250.0, freeze=True)
        base = build_source_finetuned_foundation_model(bb, 4, X_src, y_src, device=dev,
                   lr_src=1e-3, weight_decay=1e-4, max_epochs_src=200, patience_src=25,
                   val_fraction_src=0.1, batch_size=32, seed=SEED)
        for k in K_GRID:
            n_cal = minutes_to_trials(k, 4, epoch_len_sec=4.0)
            for rep in range(N_REPEATS):
                rng = np.random.default_rng(SEED + rep * 1000)
                Xc, yc, _ = sample_calibration_set(Xpool, ypool, n_cal, rng)
                model = FoundationWithHead(copy.deepcopy(base.backbone), 4).to(dev)
                model.head.load_state_dict(base.head.state_dict())
                merged = lora_adapt(model, Xc, yc, dev)
                bbk = merged.backbone.to(dev)
                f_src = extract_foundation_features(bbk, X_src, dev, 32)
                f_src, y_src_sub = _stratified_subsample(f_src, y_src, SOURCE_CAP, np.random.default_rng(SEED))
                f_cal = extract_foundation_features(bbk, Xc, dev, 32)
                f_unlab = extract_foundation_features(bbk, Xpool, dev, 32)
                f_te = extract_foundation_features(bbk, Xte, dev, 32)
                mu, sig = f_unlab.mean(0, keepdims=True), f_unlab.std(0, keepdims=True) + 1e-8
                # source convex head v_bar on LoRA-adapted source features (fixed gates G)
                key = jax.random.PRNGKey(SEED); G, key = sample_gates(f_src.shape[1], N_NEURONS, key)
                src = build_fixed_gate_model(((f_src - mu) / sig).astype(np.float32), y_src_sub, 4,
                                             N_NEURONS, BETA, RHO, key, G)
                anchored_admm(src, AP, v_anchor=None, anchor_a=0.0); v_bar = src.v
                for cfg in CONFIGS:
                    tgt = solve_target(f_src, y_src_sub, f_cal, yc, mu, sig, G, v_bar, cfg[0], cfg[1])
                    Xn = ((f_te - mu) / sig).astype(np.float32)
                    pred = np.array(tgt.stacked_predict(jnp.array(Xn), tgt.theta1, tgt.theta2)).argmax(1)
                    bca = balanced_accuracy(yte, pred)
                    scores[cfg].append(bca); per_k[cfg][k].append(bca)
        print(f"subject {subj} done", flush=True)

    print("\n=== LoRA+transfer-anchor sweep (9 subj, K=[1,10,30]) vs iter-8 0.6110 / lora 0.5955 ===")
    out = {}
    for c in sorted(CONFIGS, key=lambda c: -np.mean(scores[c])):
        s = float(np.mean(scores[c]))
        pk = {k: round(float(np.mean(per_k[c][k])), 4) for k in K_GRID}
        flag = "  <-- beats iter-8" if s > 0.6110 else ""
        print(f"  stage2={c[0]:<11} a={c[1]:<5} score={s:.4f}  per_k={pk}{flag}")
        out[f"{c[0]}_a{c[1]}"] = {"score": s, "per_k": pk}
    Path("research/runs/lora_transfer_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
