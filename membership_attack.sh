#!/usr/bin/env bash
set -euo pipefail

# LiRA-style repeated-patient FedMIA/BINN evaluation.
#
# The current default is the next ablation after the 5-client / 10-round
# baseline: more non-target clients for FedMIA's Qout estimate and more
# communication rounds for temporal evidence.
#
#   RUNS=5 ROUNDS=5 GPU=0 bash membership_attack.sh
#
# For a quick plumbing check:
#
#   RUNS=2 ROUNDS=1 LOCAL_EPOCHS=1 AUDIT_COUNT=4 MAX_SAMPLES=64 FEATURE_LIMIT=2048 DEVICE=cpu bash membership_attack.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

python3 -u experiments/fedmia_binn_lira_protocol.py \
  --runs "${RUNS:-10}" \
  --rounds "${ROUNDS:-20}" \
  --local-epochs "${LOCAL_EPOCHS:-2}" \
  --num-clients "${NUM_CLIENTS:-10}" \
  --samples-per-client "${SAMPLES_PER_CLIENT:-64}" \
  --audit-count "${AUDIT_COUNT:-32}" \
  --inclusion-prob "${INCLUSION_PROB:-0.5}" \
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
