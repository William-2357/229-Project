#!/usr/bin/env bash
# Run the full benchmark on Modal: 4 foundation + 3 specialist backbones.
#
# Same configuration as the latest run in results/foundation/, except the two
# source-anchored CLD methods (foundation_sft_anchored_cld / ..._ea_anchored_cld)
# are REPLACED by the K-adaptive anchored variants (EA + non-EA) on the foundation
# backbones. Specialists have no K-adaptive adapter, so they keep their existing
# 10-method list (anchored_cld / ea_anchored_cld).
#
# Each `modal run` is restart-safe: it resumes from modal_summary.json and
# checkpoints after every (method, subject) job. Re-run any line to continue.
#
# Prereqs (one-time): eeg-data volume must hold the raw data + 4 checkpoints, and
# the preprocessing caches must be built. See run instructions / README.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="python -m modal run modal_runner.py"

# Foundation: drop both anchored_cld, add the K-adaptive anchored variants (EA + non-EA).
FM="foundation_sft_loso,foundation_sft_ea,foundation_sft_tta,foundation_sft_finetune,foundation_sft_lora,foundation_sft_ea_lora,foundation_sft_cld,foundation_sft_ea_cld,foundation_sft_kadaptive_anchored_cld,foundation_sft_ea_kadaptive_anchored_cld"

# Specialist: no K-adaptive adapter exists — keep the existing anchored variants.
SM="loso,ea,tta,finetune,lora,ea_lora,cld,ea_cld,anchored_cld,ea_anchored_cld"

# --- Foundation backbones (each needs its matching checkpoint on /data) ---------
$RUN --backbone cbramod  --methods "$FM" --checkpoint-path /data/CBraMod_checkpoint.pth
$RUN --backbone labram   --methods "$FM" --checkpoint-path /data/labram-base.pth
$RUN --backbone mirepnet --methods "$FM" --checkpoint-path /data/MIRepNet.pth
$RUN --backbone neurogpt --methods "$FM" --checkpoint-path /data/neuro_gpt.pt

# --- Specialist backbones (no checkpoint) ---------------------------------------
$RUN --backbone eegnet      --methods "$SM"
$RUN --backbone shallowconv --methods "$SM"
$RUN --backbone conformer   --methods "$SM"

echo "All backbones dispatched. Pull results with:"
echo "  python -m modal volume get eeg-results /results ./results"
