#!/usr/bin/env bash
set -euo pipefail

# Optional sequential round-grid runner for the LiRA-style FedMIA/BINN protocol.
# The main launcher (`membership_attack.sh`) is now the preferred path for the
# serious run; this script is kept for explicit ablations only.
#
# Default grid:
#   ROUNDS=20  LOCAL_EPOCHS=2
#   ROUNDS=50  LOCAL_EPOCHS=2
#   ROUNDS=100 LOCAL_EPOCHS=2
#
# Defaults match the serious protocol except for the explicit round sweep:
#   RUNS=30, NUM_CLIENTS=10, AUDIT_COUNT=32, SAMPLES_PER_CLIENT=98
#
# Quick dry run:
#   RUNS=2 ROUND_GRID="2 3" LOCAL_EPOCHS=1 NUM_CLIENTS=5 AUDIT_COUNT=4 SAMPLES_PER_CLIENT=4 BATCH_SIZE=4 MAX_SAMPLES=64 FEATURE_LIMIT=512 DEVICE=cpu bash membership_attack_round_grid.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

RUNS="${RUNS:-30}"
ROUND_GRID="${ROUND_GRID:-20 50 100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
NUM_CLIENTS="${NUM_CLIENTS:-10}"
AUDIT_COUNT="${AUDIT_COUNT:-32}"
SAMPLES_PER_CLIENT="${SAMPLES_PER_CLIENT:-98}"

echo "FedMIA BINN round grid"
echo "  RUNS=${RUNS}"
echo "  ROUND_GRID=${ROUND_GRID}"
echo "  LOCAL_EPOCHS=${LOCAL_EPOCHS}"
echo "  NUM_CLIENTS=${NUM_CLIENTS}"
echo "  AUDIT_COUNT=${AUDIT_COUNT}"
echo "  SAMPLES_PER_CLIENT=${SAMPLES_PER_CLIENT}"
echo "  GPU=${GPU:-0}"

for ROUND_COUNT in ${ROUND_GRID}; do
  echo
  echo "=== FedMIA grid cell: ROUNDS=${ROUND_COUNT} LOCAL_EPOCHS=${LOCAL_EPOCHS} RUNS=${RUNS} ==="
  RUNS="${RUNS}" \
  ROUNDS="${ROUND_COUNT}" \
  LOCAL_EPOCHS="${LOCAL_EPOCHS}" \
  NUM_CLIENTS="${NUM_CLIENTS}" \
  AUDIT_COUNT="${AUDIT_COUNT}" \
  SAMPLES_PER_CLIENT="${SAMPLES_PER_CLIENT}" \
  bash membership_attack.sh
done
