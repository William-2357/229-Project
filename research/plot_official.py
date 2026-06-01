"""Plot the official full 9-subject convex-vs-baselines comparison (corrected 250Hz)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "research" / "runs"

def curve(tag):
    p = RUNS / f"{tag}__full.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    ks = sorted(float(k) for k in d["per_k"])
    return ks, [d["per_k"][str(k) if str(k) in d["per_k"] else f"{k:.1f}"] for k in ks]

series = [
    ("iter3_convex_full",  "convex (frozen MIRepNet + convex head)", "tab:red",   "o"),
    ("full_lora_250",      "sft_lora",                                "tab:orange","s"),
    ("full_ft_250",        "sft_finetune",                            "tab:blue",  "^"),
]
plt.figure(figsize=(7.5, 5))
for tag, label, color, mk in series:
    c = curve(tag)
    if c is None:
        continue
    ks, ys = c
    plt.plot(ks, ys, marker=mk, color=color, label=f"{label} (mean {sum(ys)/len(ys):.3f})")

plt.xscale("log")
plt.xticks([0.5,1,2,5,10,15,30], ["0.5","1","2","5","10","15","30"])
plt.xlabel("Calibration window K (minutes)")
plt.ylabel("Mean test BCA (9 subjects)")
plt.title("BCIC-IV-2a calibration: convex head vs LoRA / finetune\n(source-pretrained MIRepNet @250Hz, same backbone)")
plt.grid(True, alpha=0.3)
plt.legend()
out = REPO / "research" / "convex_vs_baselines_full.png"
plt.tight_layout(); plt.savefig(out, dpi=130)
print(f"saved {out}")
