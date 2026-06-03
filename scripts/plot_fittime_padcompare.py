"""Fit-time no-pad vs pad-256 comparison for one backbone.
Solid = padding=false, dashed = padding=256 (convex methods only). K<=15, compile-excluded.
Reads /tmp/ftc/{bb}_false.json and /tmp/ftc/{bb}_256.json (pulled from the volume)."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAD = [("Fine-tune","foundation_sft_finetune","finetune"),
        ("LoRA","foundation_sft_lora","lora"),("EA+LoRA","foundation_sft_ea_lora","ea_lora")]
CONV = [("CHA","foundation_sft_cld","cld"),("EA+CHA","foundation_sft_ea_cld","ea_cld"),
        ("A-CHA","foundation_sft_kadaptive_anchored_cld","kadaptive_anchored_cld"),
        ("EA+A-CHA","foundation_sft_ea_kadaptive_anchored_cld","ea_kadaptive_anchored_cld")]
KS=[0.5,1,2,5,10,15]


def curve(d, keys):
    perk={}
    for key,cells in d.items():
        if key.split("/")[0] not in keys: continue
        for k,r in cells.items():
            if isinstance(r,dict) and float(k) in KS:
                perk.setdefault(float(k),[]).append(r.get("fit_time_warm", r.get("fit_time")))
    xs=[k for k in KS if k in perk]; return xs,[np.mean(perk[k]) for k in xs]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--backbone",default="cbramod"); a=ap.parse_args()
    bb=a.backbone
    nopad=json.load(open(f"/tmp/ftc/{bb}_false.json"))
    try: pad=json.load(open(f"/tmp/ftc/{bb}_256.json"))
    except Exception: pad={}
    cmap=plt.get_cmap("tab10"); col={lab:cmap(i) for i,(lab,_,_) in enumerate(GRAD+CONV)}
    fig,ax=plt.subplots(figsize=(8.5,5.8))
    for lab,fk,sk in GRAD:
        xs,ys=curve(nopad,{fk,sk})
        if xs: ax.plot(xs,ys,marker="o",ms=5,lw=1.6,color=col[lab],label=lab)
    for lab,fk,sk in CONV:
        xs,ys=curve(nopad,{fk,sk})
        if xs: ax.plot(xs,ys,marker="o",ms=5,lw=1.6,color=col[lab],label=f"{lab} (no pad)")
        xs2,ys2=curve(pad,{fk,sk})
        if xs2: ax.plot(xs2,ys2,marker="s",ms=4,lw=1.4,ls="--",color=col[lab],label=f"{lab} (pad 256)")
    ax.set_xlabel("Minutes of Target Data Available",fontsize=12,fontweight="bold")
    ax.set_ylabel("Fit time (s, compile-excluded)",fontsize=12,fontweight="bold")
    ax.grid(True,alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=8,ncol=2)
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{bb}_fit_time.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved {OUT_DIR / f'{bb}_fit_time.png'}")


if __name__=="__main__":
    main()
