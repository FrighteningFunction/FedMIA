#!/usr/bin/env bash
set -euo pipefail

# Research-grade FedMIA/BINN evaluation.
#
# The default mirrors the central-model repeated-evaluation idea: evaluate the
# same patient membership task over many independently initialized federated
# BINN trajectories. The experiment reports gradient-cosine FedMIA, loss-based
# FedMIA, and their combined score. Override any knob from the shell, for example:
#
#   RUNS=5 ROUNDS=5 GPU=0 bash membership_attack.sh
#
# For a quick plumbing check:
#
#   RUNS=1 ROUNDS=1 LOCAL_EPOCHS=1 MAX_SAMPLES=64 FEATURE_LIMIT=2048 DEVICE=cpu bash membership_attack.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

python3 -u experiments/fedmia_binn_research.py \
  --runs "${RUNS:-200}" \
  --rounds "${ROUNDS:-20}" \
  --local-epochs "${LOCAL_EPOCHS:-2}" \
  --num-clients "${NUM_CLIENTS:-5}" \
  --samples-per-client "${SAMPLES_PER_CLIENT:-64}" \
  --candidate-count "${CANDIDATE_COUNT:-32}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --lr "${LR:-0.03}" \
  --threshold "${THRESHOLD:-0.5}" \
  --threshold-grid "${THRESHOLD_GRID:-0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90}" \
  --seed "${SEED:-20260509}" \
  --device "${DEVICE:-cuda}" \
  --gpu "${GPU:-0}" \
  --x-file "${X_FILE:-pnet_x.npy}" \
  --y-file "${Y_FILE:-pnet_y.npy}" \
  --max-samples "${MAX_SAMPLES:-0}" \
  --feature-limit "${FEATURE_LIMIT:-0}" \
  --log-dir "${LOG_DIR:-logs}" \
  --report-dir "${REPORT_DIR:-reports}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  "$@"
