#!/usr/bin/env bash
set -euo pipefail

# FedMIA paper-style BINN grid evaluation.
#
# This launcher intentionally follows the FedMIA paper baseline shape:
# FedAvg, one target client, non-target clients estimate Qout, FedMIA-I uses
# loss, FedMIA-II uses gradient cosine, and the grid axes are clients,
# communication rounds, local epochs, sample volume, and IID/Dirichlet beta.
#
#   GPU=0 bash membership_attack.sh
#
# Paper-style ablation example:
#
#   CLIENT_GRID="5,10,20,30" ROUND_GRID="100,200,300" LOCAL_EPOCH_GRID="1,3,5,9" BETA_GRID="iid,10,1,0.1" GPU=0 bash membership_attack.sh
#
# For a quick plumbing check:
#
#   PLUMBING=1 bash membership_attack.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

if [[ "${PLUMBING:-0}" == "1" ]]; then
  RUNS_DEFAULT=1
  CLIENT_GRID_DEFAULT=5
  ROUND_GRID_DEFAULT=1
  LOCAL_EPOCH_GRID_DEFAULT=1
  BETA_GRID_DEFAULT=iid
  SAMPLE_FRACTION_GRID_DEFAULT=1.0
  SAMPLES_PER_CLIENT_GRID_DEFAULT=8
  CANDIDATE_COUNT_DEFAULT=4
  MAX_SAMPLES_DEFAULT=80
  FEATURE_LIMIT_DEFAULT=256
  DEVICE_DEFAULT=cpu
  MAX_CONFIGS_DEFAULT=1
else
  RUNS_DEFAULT=3
  CLIENT_GRID_DEFAULT=10
  ROUND_GRID_DEFAULT=300
  LOCAL_EPOCH_GRID_DEFAULT=1
  BETA_GRID_DEFAULT=iid
  SAMPLE_FRACTION_GRID_DEFAULT=1.0
  SAMPLES_PER_CLIENT_GRID_DEFAULT=auto
  CANDIDATE_COUNT_DEFAULT=0
  MAX_SAMPLES_DEFAULT=0
  FEATURE_LIMIT_DEFAULT=0
  DEVICE_DEFAULT=cuda
  MAX_CONFIGS_DEFAULT=0
fi

python3 -u experiments/fedmia_binn_paper_grid.py \
  --runs "${RUNS:-${RUNS_DEFAULT}}" \
  --client-grid "${CLIENT_GRID:-${NUM_CLIENTS:-${CLIENT_GRID_DEFAULT}}}" \
  --round-grid "${ROUND_GRID:-${ROUNDS:-${ROUND_GRID_DEFAULT}}}" \
  --local-epoch-grid "${LOCAL_EPOCH_GRID:-${LOCAL_EPOCHS:-${LOCAL_EPOCH_GRID_DEFAULT}}}" \
  --beta-grid "${BETA_GRID:-${BETA_GRID_DEFAULT}}" \
  --sample-fraction-grid "${SAMPLE_FRACTION_GRID:-${SAMPLE_FRACTION_GRID_DEFAULT}}" \
  --samples-per-client-grid "${SAMPLES_PER_CLIENT_GRID:-${SAMPLES_PER_CLIENT:-${SAMPLES_PER_CLIENT_GRID_DEFAULT}}}" \
  --candidate-count "${CANDIDATE_COUNT:-${CANDIDATE_COUNT_DEFAULT}}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --lr "${LR:-0.03}" \
  --weight-decay "${WEIGHT_DECAY:-5e-4}" \
  --momentum "${MOMENTUM:-0.9}" \
  --threshold "${THRESHOLD:-0.5}" \
  --threshold-grid "${THRESHOLD_GRID:-0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90}" \
  --holdout-fraction "${HOLDOUT_FRACTION:-0.1}" \
  --seed "${SEED:-20260509}" \
  --device "${DEVICE:-${DEVICE_DEFAULT}}" \
  --gpu "${GPU:-0}" \
  --x-file "${X_FILE:-pnet_x.npy}" \
  --y-file "${Y_FILE:-pnet_y.npy}" \
  --max-samples "${MAX_SAMPLES:-${MAX_SAMPLES_DEFAULT}}" \
  --feature-limit "${FEATURE_LIMIT:-${FEATURE_LIMIT_DEFAULT}}" \
  --max-configs "${MAX_CONFIGS:-${MAX_CONFIGS_DEFAULT}}" \
  --log-dir "${LOG_DIR:-logs}" \
  --report-dir "${REPORT_DIR:-reports}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  "$@"
