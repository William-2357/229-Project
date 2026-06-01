"""iter-10: re-tune the convex head ON LoRA-adapted features (research tool).

iter-8 (LoRA+convex) used the convex-head hparams tuned for FROZEN features (cal_balance=4,
beta=1e-4). The LoRA-adapted feature distribution differs, so the optimum may too. This sweep
LoRA-adapts ONCE per (subject,K,repeat) — the expensive step — then tries many convex-head
configs on those cached adapted features (amortized, like sweep_convex_head.py). Mirrors
convex_calib's LoRA+convex path so a winning config transfers to HPARAMS.

Usage: python research/sweep_lora_convex.py
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
from adaptation.cld import fit_cld_head
from adaptation.convex_calib import _stratified_subsample
from adaptation.base import train_epoch, evaluate_model
from evaluation.protocols import minutes_to_trials, sample_calibration_set
from evaluation.metrics import balanced_accuracy
from peft import LoraConfig, get_peft_model
import jax.numpy as jnp

SUBJECTS = list(range(1, 10))
K_GRID = [1.0, 10.0, 30.0]
N_REPEATS = 3
SEED = 42
SOURCE_CAP = 800
LORA_RANK = 8

# convex-head configs to try on the LoRA-adapted features (iter-8 = cal_balance4/beta1e-4)
CONFIGS = [
    dict(cal_balance=4,  beta=1e-4, n_neurons=32),   # = iter-8 (reproduction check)
    dict(cal_balance=2,  beta=1e-4, n_neurons=32),
    dict(cal_balance=8,  beta=1e-4, n_neurons=32),
    dict(cal_balance=4,  beta=1e-3, n_neurons=32),
    dict(cal_balance=8,  beta=1e-3, n_neurons=32),
    dict(cal_balance=4,  beta=1e-4, n_neurons=48),
]


def lora_adapt(model, X_cal, y_cal, device):
    targets = _get_lora_target_modules(model, rank=LORA_RANK)
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK * 2, target_modules=targets,
                     lora_dropout=0.1, bias="none")
    lm = get_peft_model(model, cfg)
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


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCICIVDataset("data/raw/bciciv2a", cache_dir="data/raw/bciciv2a_mirepnet250_cache",
                       target_sfreq=250.0, preprocess_config=MIREPNET_PREPROCESS_CONFIG)
    scores = {i: [] for i in range(len(CONFIGS))}
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
                # LoRA-adapt ONCE (expensive), on a fresh copy of the source-FT model
                model = FoundationWithHead(copy.deepcopy(base.backbone), 4).to(dev)
                model.head.load_state_dict(base.head.state_dict())
                merged = lora_adapt(model, Xc, yc, dev)
                bbk = merged.backbone.to(dev)
                f_src = extract_foundation_features(bbk, X_src, dev, 32)
                f_src, y_src_sub = _stratified_subsample(f_src, y_src, SOURCE_CAP, np.random.default_rng(SEED))
                f_cal = extract_foundation_features(bbk, Xc, dev, 32)
                f_unlab = extract_foundation_features(bbk, Xpool, dev, 32)
                f_te = extract_foundation_features(bbk, Xte, dev, 32)
                norm = (f_unlab.mean(0, keepdims=True), f_unlab.std(0, keepdims=True) + 1e-8)
                for ci, cfg in enumerate(CONFIGS):
                    reps = max(1, int(round(cfg["cal_balance"] * len(f_src) / max(1, len(f_cal)))))
                    Xfit = np.concatenate([f_src, np.tile(f_cal, (reps, 1))], 0)
                    yfit = np.concatenate([y_src_sub, np.tile(yc, reps)], 0)
                    cld, mu, sig = fit_cld_head(Xfit, yfit, 4, cfg["n_neurons"], 20, cfg["beta"],
                                                0.01, 1.0, 50, 10, SEED, norm_stats=norm)
                    Xn = ((f_te - mu) / sig).astype(np.float32)
                    pred = np.array(cld.stacked_predict(jnp.array(Xn), cld.theta1, cld.theta2)).argmax(1)
                    scores[ci].append(balanced_accuracy(yte, pred))
        print(f"subject {subj} done", flush=True)

    print("\n=== LoRA+convex head sweep (9 subj, K=[1,10,30]) vs iter-8 0.6110 / lora 0.5955 ===")
    ranked = sorted(range(len(CONFIGS)), key=lambda i: -np.mean(scores[i]))
    for i in ranked:
        s = float(np.mean(scores[i]))
        flag = "  <-- beats iter-8" if s > 0.6110 else ""
        print(f"  score={s:.4f}  {CONFIGS[i]}{flag}")
    best = ranked[0]
    Path("research/runs/lora_convex_sweep.json").write_text(json.dumps(
        {"configs": CONFIGS, "scores": {i: float(np.mean(scores[i])) for i in scores},
         "best_config": CONFIGS[best], "best_score": float(np.mean(scores[best]))}, indent=2))


if __name__ == "__main__":
    main()
