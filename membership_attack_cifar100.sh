#!/usr/bin/env bash
set -euo pipefail

# FedMIA-I/FedMIA-II reproduction for AlexNet on CIFAR100.
#
# Full paper-style grid requested for the current reproduction:
#
#   RUNS=10
#   CLIENT_GRID="5,10"
#   ROUND_GRID="100,200,300"
#   LOCAL_EPOCH_GRID="1,3,5"
#   BETA_GRID="iid,10,1,0.1"
#
# CIFAR100 must exist under data/datasets/CIFAR100/cifar-100-python, or set
# DOWNLOAD=1 to fetch the official archive.
#
# Plumbing check:
#
#   PLUMBING=1 bash membership_attack_cifar100.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

if [[ "${PLUMBING:-0}" == "1" ]]; then
  RUNS_DEFAULT=1
  CLIENT_GRID_DEFAULT=5
  ROUND_GRID_DEFAULT=1
  LOCAL_EPOCH_GRID_DEFAULT=1
  BETA_GRID_DEFAULT=iid
  SAMPLE_FRACTION_GRID_DEFAULT=1.0
  SAMPLES_PER_CLIENT_GRID_DEFAULT=20
  CANDIDATE_COUNT_DEFAULT=4
  MAX_TRAIN_SAMPLES_DEFAULT=200
  MAX_TEST_SAMPLES_DEFAULT=50
  MAX_CONFIGS_DEFAULT=1
  DEVICE_DEFAULT=cpu
  SYNTHETIC_FLAG_DEFAULT=1
else
  RUNS_DEFAULT=10
  CLIENT_GRID_DEFAULT="5,10"
  ROUND_GRID_DEFAULT="100,200,300"
  LOCAL_EPOCH_GRID_DEFAULT="1,3,5"
  BETA_GRID_DEFAULT="iid,10,1,0.1"
  SAMPLE_FRACTION_GRID_DEFAULT=1.0
  SAMPLES_PER_CLIENT_GRID_DEFAULT=auto
  CANDIDATE_COUNT_DEFAULT=512
  MAX_TRAIN_SAMPLES_DEFAULT=0
  MAX_TEST_SAMPLES_DEFAULT=0
  MAX_CONFIGS_DEFAULT=0
  DEVICE_DEFAULT=cuda
  SYNTHETIC_FLAG_DEFAULT=0
fi

EXTRA_ARGS=()
if [[ "${DOWNLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--download)
fi
if [[ "${SYNTHETIC_DATA:-${SYNTHETIC_FLAG_DEFAULT}}" == "1" ]]; then
  EXTRA_ARGS+=(--synthetic-data)
fi
if [[ "${LOG_CLIENT_EPOCHS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-client-epochs)
fi

python3 -u experiments/fedmia_cifar100_alexnet_grid.py \
  --runs "${RUNS:-${RUNS_DEFAULT}}" \
  --client-grid "${CLIENT_GRID:-${CLIENT_GRID_DEFAULT}}" \
  --round-grid "${ROUND_GRID:-${ROUND_GRID_DEFAULT}}" \
  --local-epoch-grid "${LOCAL_EPOCH_GRID:-${LOCAL_EPOCH_GRID_DEFAULT}}" \
  --beta-grid "${BETA_GRID:-${BETA_GRID_DEFAULT}}" \
  --sample-fraction-grid "${SAMPLE_FRACTION_GRID:-${SAMPLE_FRACTION_GRID_DEFAULT}}" \
  --samples-per-client-grid "${SAMPLES_PER_CLIENT_GRID:-${SAMPLES_PER_CLIENT:-${SAMPLES_PER_CLIENT_GRID_DEFAULT}}}" \
  --candidate-count "${CANDIDATE_COUNT:-${CANDIDATE_COUNT_DEFAULT}}" \
  --attack-every "${ATTACK_EVERY:-1}" \
  --nonmember-other-client-fraction "${NONMEMBER_OTHER_CLIENT_FRACTION:-0.1}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --lr "${LR:-0.1}" \
  --lr-decay "${LR_DECAY:-0.99}" \
  --weight-decay "${WEIGHT_DECAY:-5e-4}" \
  --momentum "${MOMENTUM:-0.9}" \
  --threshold "${THRESHOLD:-0.5}" \
  --threshold-grid "${THRESHOLD_GRID:-0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90}" \
  --seed "${SEED:-20260511}" \
  --device "${DEVICE:-${DEVICE_DEFAULT}}" \
  --gpu "${GPU:-0}" \
  --data-dir "${DATA_DIR:-data/datasets/CIFAR100}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-${MAX_TRAIN_SAMPLES_DEFAULT}}" \
  --max-test-samples "${MAX_TEST_SAMPLES:-${MAX_TEST_SAMPLES_DEFAULT}}" \
  --max-configs "${MAX_CONFIGS:-${MAX_CONFIGS_DEFAULT}}" \
  --log-dir "${LOG_DIR:-logs}" \
  --report-dir "${REPORT_DIR:-reports}" \
  --log-level "${LOG_LEVEL:-INFO}" \
  --num-workers "${NUM_WORKERS:-2}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
