#!/usr/bin/env bash
# Run ONLY the K-adaptive source-anchored CLD methods on Modal, for every backbone.
#
# This is the "anchored methods only" run: instead of the usual fixed-anchor
# anchored_cld, every backbone is evaluated with the K-ADAPTIVE explicit-anchor
# variant (EA + non-EA). Results merge into the existing per-backbone
# modal_summary.json (orchestrator resumes + checkpoints after every job), so they
# sit alongside results/foundation/ and results/specialist/ from the main run.
#
#   Foundations -> foundation_sft_kadaptive_anchored_cld (+ EA)
#   Specialists -> kadaptive_anchored_cld               (+ EA)   [new adapter]
#
# Each `modal run` is restart-safe: re-run any line to continue where it stopped.
#
# Prereqs (one-time): eeg-data volume must hold the raw npz data + the 3 foundation
# checkpoints, and the preprocessing caches must be built.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
# For long multi-backbone runs prefer `--detach` (survives client disconnect) and
# poll the eeg-results volume for completion. For shorter runs the blocking form
# below streams live progress and returns a real completion signal.
RUN="$PY -m modal run modal_runner.py"

FM="foundation_sft_kadaptive_anchored_cld,foundation_sft_ea_kadaptive_anchored_cld"
SM="kadaptive_anchored_cld,ea_kadaptive_anchored_cld"

# --- Foundation backbones (each needs its matching checkpoint on /data) ---------
$RUN --backbone labram   --methods "$FM" --checkpoint-path /data/labram-base.pth
$RUN --backbone mirepnet --methods "$FM" --checkpoint-path /data/MIRepNet.pth
$RUN --backbone neurogpt --methods "$FM" --checkpoint-path /data/neuro_gpt.pt

# --- Specialist backbones (no checkpoint) ---------------------------------------
$RUN --backbone eegnet      --methods "$SM"
$RUN --backbone shallowconv --methods "$SM"
$RUN --backbone conformer   --methods "$SM"

echo "All K-adaptive anchored runs dispatched. Pull results with:"
echo "  python -m modal volume get eeg-results /results ./results"
