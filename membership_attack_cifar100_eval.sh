#!/usr/bin/env bash
set -euo pipefail

# Compact AlexNet/CIFAR100 FedMIA evaluation.
#
# This is intentionally NOT a full factorial grid. Each line below is one
# concrete configuration from the FedMIA paper-style grid, repeated with
# RUNS_PER_CONFIG seeds so we can report mean/std within that configuration.
#
# Statistical note:
#   Do not average these four configurations together as if they were one
#   experiment. Report each configuration's mean/std separately. If one setting
#   matters most, rerun that setting with RUNS_PER_CONFIG=10 or more.
#
# Default run:
#   GPU=0 bash memebership_attack_cifar100_eval.sh
#
# Faster triage:
#   RUNS_PER_CONFIG=3 CANDIDATE_COUNT=256 GPU=0 bash memebership_attack_cifar100_eval.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-5}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-512}"
SAMPLE_FRACTION_GRID="${SAMPLE_FRACTION_GRID:-1.0}"
DOWNLOAD="${DOWNLOAD:-0}"
GPU="${GPU:-0}"

run_config() {
  local name="$1"
  local clients="$2"
  local rounds="$3"
  local local_epochs="$4"
  local beta="$5"

  echo
  echo "=== AlexNet/CIFAR100 FedMIA config: ${name} ==="
  echo "  RUNS=${RUNS_PER_CONFIG}"
  echo "  CLIENT_GRID=${clients}"
  echo "  ROUND_GRID=${rounds}"
  echo "  LOCAL_EPOCH_GRID=${local_epochs}"
  echo "  BETA_GRID=${beta}"
  echo "  SAMPLE_FRACTION_GRID=${SAMPLE_FRACTION_GRID}"
  echo "  CANDIDATE_COUNT=${CANDIDATE_COUNT}"

  DOWNLOAD="${DOWNLOAD}" \
  RUNS="${RUNS_PER_CONFIG}" \
  CLIENT_GRID="${clients}" \
  ROUND_GRID="${rounds}" \
  LOCAL_EPOCH_GRID="${local_epochs}" \
  BETA_GRID="${beta}" \
  SAMPLE_FRACTION_GRID="${SAMPLE_FRACTION_GRID}" \
  CANDIDATE_COUNT="${CANDIDATE_COUNT}" \
  GPU="${GPU}" \
  bash membership_attack_cifar100.sh
}

# Baseline-ish short setting: paper default client count, shorter 100-round run.
run_config "baseline_10c_100r_1e_iid" 10 100 2 iid

# Temporal evidence check without 200 rounds.
run_config "longer_rounds_10c_200r_1e_iid" 10 200 1 iid

# Stronger memorization/leakage stressor: more local epochs plus severe Non-IID.
run_config "stress_10c_100r_5e_beta01" 10 100 5 0.1
