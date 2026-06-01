"""Compile all results into the k-minutes BCA graph (full 9-subject, corrected 250Hz)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "research" / "runs"


def curve(tag):
    """Return (ks, ys, is_full). Prefer the full 7-K sweep; fall back to the 9-subj proxy."""
    for mode in ("full", "proxy"):
        p = RUNS / f"{tag}__{mode}.json"
        if p.exists():
            d = json.loads(p.read_text())
            ks = sorted(float(k) for k in d["per_k"])
            ys = [d["per_k"][next(kk for kk in d["per_k"] if float(kk) == k)] for k in ks]
            return ks, ys, mode == "full"
    return None


series = [
    ("iter8_lora_convex_full", "iter8_lora_convex",  "LoRA + convex head  (ours, hybrid)", "tab:green",  "D", 2.4),
    ("full_lora_250",          None,                 "sft_lora",                           "tab:orange", "s", 1.8),
    ("iter3_convex_full",      None,                 "convex head (frozen backbone)",      "tab:red",    "o", 1.8),
    ("full_ft_250",            None,                 "sft_finetune",                       "tab:blue",   "^", 1.8),
]

plt.figure(figsize=(8, 5.2))
for full_tag, proxy_tag, label, color, mk, lw in series:
    c = curve(full_tag) or (curve(proxy_tag) if proxy_tag else None)
    if c is None:
        continue
    ks, ys, is_full = c
    style = "-" if is_full else "--"
    suffix = "" if is_full else "  [proxy K only — full computing]"
    plt.plot(ks, ys, style, marker=mk, color=color, linewidth=lw, markersize=6,
             label=f"{label}  (mean {sum(ys)/len(ys):.3f}){suffix}")

plt.xscale("log")
plt.xticks([0.5, 1, 2, 5, 10, 15, 30], ["0.5", "1", "2", "5", "10", "15", "30"])
plt.xlabel("Calibration window  K  (minutes of labeled target data)")
plt.ylabel("Mean test balanced accuracy (9 subjects)")
plt.title("BCIC-IV-2a calibration efficiency — convex head vs LoRA / finetune\n"
          "source-pretrained MIRepNet @250Hz (identical frozen-after-source-FT backbone)")
plt.grid(True, alpha=0.3, which="both")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
out = REPO / "research" / "kmin_results.png"
plt.savefig(out, dpi=140)
print(f"saved {out}")
