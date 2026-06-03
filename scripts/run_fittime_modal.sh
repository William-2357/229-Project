#!/usr/bin/env bash
# Fit-time benchmark: K-spanning adaptation methods, all 7 backbones, 3 subjects.
# padding=false for the convex (CLD) solves (CLD_NO_PAD=1), and fit_time_warm in the
# results excludes the JAX compile (mean over warm repeats). Writes to
# modal_summary_fittime.json (non-destructive; old 256-padding outputs cached as *_pad256).
set -euo pipefail
cd "$(dirname "$0")/.."

export CLD_NO_PAD=1          # padding=false for convex solves
export RESULTS_TAG=_fittime  # fresh result file
PY="${PY:-.venv/bin/python}"
RUN="$PY -m modal run modal_runner.py"
NS=3

FM="foundation_sft_finetune,foundation_sft_lora,foundation_sft_ea_lora,foundation_sft_cld,foundation_sft_ea_cld,foundation_sft_kadaptive_anchored_cld,foundation_sft_ea_kadaptive_anchored_cld"
SM="finetune,lora,ea_lora,cld,ea_cld,kadaptive_anchored_cld,ea_kadaptive_anchored_cld"

$RUN --backbone cbramod  --methods "$FM" --checkpoint-path /data/CBraMod_checkpoint.pth --n-subjects $NS
$RUN --backbone labram   --methods "$FM" --checkpoint-path /data/labram-base.pth        --n-subjects $NS
$RUN --backbone mirepnet --methods "$FM" --checkpoint-path /data/MIRepNet.pth           --n-subjects $NS
$RUN --backbone neurogpt --methods "$FM" --checkpoint-path /data/neuro_gpt.pt           --n-subjects $NS
$RUN --backbone eegnet      --methods "$SM" --n-subjects $NS
$RUN --backbone shallowconv --methods "$SM" --n-subjects $NS
$RUN --backbone conformer   --methods "$SM" --n-subjects $NS

echo "ALL FIT-TIME RUNS DONE"
