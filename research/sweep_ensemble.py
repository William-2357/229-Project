"""iter-18: CONTROLLED convex-head ensemble-size test (research tool).

iter-17 (M=3 ensemble) gave 0.6129 vs iter-8 0.611 — within proxy noise from independent runs.
This isolates the ensemble effect: per (subj,K,rep) fit M_MAX LoRA+convex members ONCE, store each
member's test probabilities, then report BCA for cumulative ensemble sizes {1,3,5} on the SAME
members. Same cells + same members => the size-1 vs size-M gap is the pure ensemble effect, free of
run-to-run noise. (size=1 == iter-8 LoRA+convex.)

Usage: python research/sweep_ensemble.py
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
N_NEURONS, BETA, RHO, RANK, CAL_BALANCE = 32, 1e-4, 0.01, 20, 4.0
M_MAX = 5
ENS_SIZES = [1, 3, 5]


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


def member_probs(base, X_src, y_src, Xc, yc, Xpool, Xte, dev, ms):
    torch.manual_seed(ms); np.random.seed(ms)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(ms)
    model = FoundationWithHead(copy.deepcopy(base.backbone), 4).to(dev)
    model.head.load_state_dict(base.head.state_dict())
    bbk = lora_adapt(model, Xc, yc, dev).backbone.to(dev)
    f_src = extract_foundation_features(bbk, X_src, dev, 32)
    f_src, y_sub = _stratified_subsample(f_src, y_src, SOURCE_CAP, np.random.default_rng(SEED))
    f_cal = extract_foundation_features(bbk, Xc, dev, 32)
    f_unlab = extract_foundation_features(bbk, Xpool, dev, 32)
    f_te = extract_foundation_features(bbk, Xte, dev, 32)
    mu, sig = f_unlab.mean(0, keepdims=True), f_unlab.std(0, keepdims=True) + 1e-8
    reps = max(1, int(round(CAL_BALANCE * len(f_src) / max(1, len(f_cal)))))
    Xfit = np.concatenate([f_src, np.tile(f_cal, (reps, 1))], 0)
    yfit = np.concatenate([y_sub, np.tile(yc, reps)], 0)
    cld, m_, s_ = fit_cld_head(Xfit, yfit, 4, N_NEURONS, RANK, BETA, RHO, 1.0, 50, 10, ms, norm_stats=(mu, sig))
    Xn = ((f_te - m_) / s_).astype(np.float32)
    lg = np.array(cld.stacked_predict(jnp.array(Xn), cld.theta1, cld.theta2))
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = BCICIVDataset("data/raw/bciciv2a", cache_dir="data/raw/bciciv2a_mirepnet250_cache",
                       target_sfreq=250.0, preprocess_config=MIREPNET_PREPROCESS_CONFIG)
    per_k = {sz: {k: [] for k in K_GRID} for sz in ENS_SIZES}
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
                probs = [member_probs(base, X_src, y_src, Xc, yc, Xpool, Xte, dev, SEED + 1000 * (m + 1))
                         for m in range(M_MAX)]
                for sz in ENS_SIZES:
                    avg = np.mean(probs[:sz], axis=0)
                    per_k[sz][k].append(balanced_accuracy(yte, avg.argmax(1)))
        print(f"subject {subj} done", flush=True)

    print("\n=== CONTROLLED ensemble-size test (9 subj, K=[1,10,30], same members) ===")
    print("    (size=1 == iter-8 LoRA+convex; gap to size>1 is the pure ensemble effect)")
    out = {}
    for sz in ENS_SIZES:
        pk = {k: round(float(np.mean(per_k[sz][k])), 4) for k in K_GRID}
        score = float(np.mean([np.mean(per_k[sz][k]) for k in K_GRID]))
        print(f"  M={sz}: score={score:.4f}  per_k={pk}")
        out[f"M{sz}"] = {"score": score, "per_k": pk}
    Path("research/runs/ensemble_size_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
