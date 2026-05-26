#!/usr/bin/env bash
set -euo pipefail

# Compact BINN/FedMIA paper-style evaluation.
#
# This launcher is intentionally NOT a full factorial grid. It runs five
# concrete configurations from the FedMIA paper's setting ranges, using the
# existing Python implementation through membership_attack.sh.
#
# Default run:
#   GPU=0 bash membership_attack_binn_eval.sh
#
# Faster triage:
#   RUNS_PER_CONFIG=1 CANDIDATE_COUNT=128 GPU=0 bash membership_attack_binn_eval.sh
#
# Plumbing check:
#   PLUMBING=1 bash membership_attack_binn_eval.sh
#
# Stronger within-config uncertainty:
#   RUNS_PER_CONFIG=3 GPU=0 bash membership_attack_binn_eval.sh

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

if [[ "${PLUMBING:-0}" == "1" ]]; then
  RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-1}"
  SAMPLE_FRACTION_GRID="${SAMPLE_FRACTION_GRID:-1.0}"
  SAMPLES_PER_CLIENT_GRID="${SAMPLES_PER_CLIENT_GRID:-8}"
  CANDIDATE_COUNT="${CANDIDATE_COUNT:-4}"
  BATCH_SIZE="${BATCH_SIZE:-4}"
  LR="${LR:-0.03}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
  MOMENTUM="${MOMENTUM:-0.9}"
  THRESHOLD="${THRESHOLD:-0.5}"
  HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.1}"
  DEVICE="${DEVICE:-cpu}"
  MAX_SAMPLES="${MAX_SAMPLES:-80}"
  FEATURE_LIMIT="${FEATURE_LIMIT:-256}"
else
  RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-1}"
  SAMPLE_FRACTION_GRID="${SAMPLE_FRACTION_GRID:-1.0}"
  SAMPLES_PER_CLIENT_GRID="${SAMPLES_PER_CLIENT_GRID:-auto}"
  CANDIDATE_COUNT="${CANDIDATE_COUNT:-0}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  LR="${LR:-0.03}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
  MOMENTUM="${MOMENTUM:-0.9}"
  THRESHOLD="${THRESHOLD:-0.5}"
  HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.1}"
  DEVICE="${DEVICE:-cuda}"
  MAX_SAMPLES="${MAX_SAMPLES:-0}"
  FEATURE_LIMIT="${FEATURE_LIMIT:-0}"
fi

GPU="${GPU:-0}"
REPORT_DIR="${REPORT_DIR:-reports}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

EVAL_ID="fedmia_binn_eval_$(date +%Y-%m-%d-%H-%M)_$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
CSV_LIST_FILE="$(mktemp)"
trap 'rm -f "${CSV_LIST_FILE}"' EXIT

run_config() {
  local name="$1"
  local clients="$2"
  local rounds="$3"
  local local_epochs="$4"
  local beta="$5"

  echo
  echo "=== BINN FedMIA eval config: ${name} ==="
  echo "  RUNS=${RUNS_PER_CONFIG}"
  echo "  CLIENT_GRID=${clients}"
  echo "  ROUND_GRID=${rounds}"
  echo "  LOCAL_EPOCH_GRID=${local_epochs}"
  echo "  BETA_GRID=${beta}"
  echo "  SAMPLE_FRACTION_GRID=${SAMPLE_FRACTION_GRID}"
  echo "  SAMPLES_PER_CLIENT_GRID=${SAMPLES_PER_CLIENT_GRID}"
  echo "  CANDIDATE_COUNT=${CANDIDATE_COUNT}"

  RUNS="${RUNS_PER_CONFIG}" \
  CLIENT_GRID="${clients}" \
  ROUND_GRID="${rounds}" \
  LOCAL_EPOCH_GRID="${local_epochs}" \
  BETA_GRID="${beta}" \
  SAMPLE_FRACTION_GRID="${SAMPLE_FRACTION_GRID}" \
  SAMPLES_PER_CLIENT_GRID="${SAMPLES_PER_CLIENT_GRID}" \
  CANDIDATE_COUNT="${CANDIDATE_COUNT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LR="${LR}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  MOMENTUM="${MOMENTUM}" \
  THRESHOLD="${THRESHOLD}" \
  HOLDOUT_FRACTION="${HOLDOUT_FRACTION}" \
  DEVICE="${DEVICE}" \
  GPU="${GPU}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  FEATURE_LIMIT="${FEATURE_LIMIT}" \
  REPORT_DIR="${REPORT_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  bash membership_attack.sh

  local latest_csv
  latest_csv="$(ls -t "${REPORT_DIR}"/fedmia_binn_paper_grid_*.csv | head -1)"
  echo "${latest_csv}" >> "${CSV_LIST_FILE}"
}

if [[ "${PLUMBING:-0}" == "1" ]]; then
  run_config "plumbing_5c_1r_1e_iid" 5 1 1 iid
  run_config "plumbing_5c_2r_1e_iid" 5 2 1 iid
  run_config "plumbing_5c_1r_2e_iid" 5 1 2 iid
  run_config "plumbing_5c_1r_1e_beta1" 5 1 1 1
  run_config "plumbing_5c_1r_1e_beta01" 5 1 1 0.1
else
  # Current representative baseline, repeated so the eval report has the exact
  # point of comparison from the latest BINN paper-grid result.
  run_config "baseline_10c_100r_2e_iid" 10 100 2 iid

  # More communication rounds: tests whether FedMIA benefits from longer update
  # history without changing the client count or local training strength.
  run_config "rounds_10c_200r_2e_iid" 10 200 2 iid

  # More local training: tests stronger client-side memorization pressure while
  # keeping the same 100 communication rounds.
  run_config "epochs_10c_100r_5e_iid" 10 100 5 iid

  # Moderate Non-IID from the paper range.
  run_config "moderate_non_iid_10c_100r_2e_beta1" 10 100 2 1

  # Paper-range stress point: fewer clients and severe Non-IID.
  run_config "stress_5c_100r_2e_beta01" 5 100 2 0.1
fi

EVAL_REPORT="${REPORT_DIR}/${EVAL_ID}.txt"
EVAL_CSV="${REPORT_DIR}/${EVAL_ID}.csv"

CSV_LIST_FILE="${CSV_LIST_FILE}" \
EVAL_REPORT="${EVAL_REPORT}" \
EVAL_CSV="${EVAL_CSV}" \
EVAL_ID="${EVAL_ID}" \
RUNS_PER_CONFIG="${RUNS_PER_CONFIG}" \
python3 - <<'PY'
import csv
import math
import os
from pathlib import Path


def as_float(row, key):
    value = row.get(key, "")
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def clean(values):
    return [value for value in values if not math.isnan(value)]


def mean(values):
    values = clean(values)
    if not values:
        return math.nan
    return sum(values) / len(values)


def sample_std(values):
    values = clean(values)
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def fmt(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


list_file = Path(os.environ["CSV_LIST_FILE"])
csv_paths = [Path(line.strip()) for line in list_file.read_text().splitlines() if line.strip()]
rows = []
for path in csv_paths:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["source_csv"] = str(path)
            rows.append(row)

selected = [
    ("train_acc_final", "train_acc_final_mean"),
    ("holdout_acc_final", "holdout_acc_final_mean"),
    ("fedmia_i_loss_auc", "fedmia_i_loss_auc_mean"),
    ("fedmia_i_loss_f1", "fedmia_i_loss_f1_mean"),
    ("fedmia_i_loss_tpr", "fedmia_i_loss_tpr_mean"),
    ("fedmia_i_loss_tnr", "fedmia_i_loss_tnr_mean"),
    ("fedmia_i_loss_fpr", "fedmia_i_loss_fpr_mean"),
    ("fedmia_i_loss_tpr_at_fpr_0.001", "fedmia_i_loss_tpr_at_fpr_0.001_mean"),
    ("fedmia_ii_cosine_auc", "fedmia_ii_cosine_auc_mean"),
    ("fedmia_ii_cosine_f1", "fedmia_ii_cosine_f1_mean"),
    ("fedmia_ii_cosine_tpr", "fedmia_ii_cosine_tpr_mean"),
    ("fedmia_ii_cosine_tnr", "fedmia_ii_cosine_tnr_mean"),
    ("fedmia_ii_cosine_fpr", "fedmia_ii_cosine_fpr_mean"),
    ("fedmia_ii_cosine_tpr_at_fpr_0.001", "fedmia_ii_cosine_tpr_at_fpr_0.001_mean"),
]

pooled_selected = [
    ("fedmia_i_loss_pooled_auc", "fedmia_i_loss_pooled_auc"),
    ("fedmia_i_loss_pooled_f1", "fedmia_i_loss_pooled_f1"),
    ("fedmia_i_loss_pooled_tpr", "fedmia_i_loss_pooled_tpr"),
    ("fedmia_i_loss_pooled_tnr", "fedmia_i_loss_pooled_tnr"),
    ("fedmia_i_loss_pooled_fpr", "fedmia_i_loss_pooled_fpr"),
    ("fedmia_i_loss_pooled_tpr_at_fpr_0.001", "fedmia_i_loss_pooled_tpr_at_fpr_0.001"),
    ("fedmia_ii_cosine_pooled_auc", "fedmia_ii_cosine_pooled_auc"),
    ("fedmia_ii_cosine_pooled_f1", "fedmia_ii_cosine_pooled_f1"),
    ("fedmia_ii_cosine_pooled_tpr", "fedmia_ii_cosine_pooled_tpr"),
    ("fedmia_ii_cosine_pooled_tnr", "fedmia_ii_cosine_pooled_tnr"),
    ("fedmia_ii_cosine_pooled_fpr", "fedmia_ii_cosine_pooled_fpr"),
    ("fedmia_ii_cosine_pooled_tpr_at_fpr_0.001", "fedmia_ii_cosine_pooled_tpr_at_fpr_0.001"),
]

eval_csv = Path(os.environ["EVAL_CSV"])
csv_columns = [
    "source_csv",
    "config_id",
    "clients",
    "rounds",
    "local_epochs",
    "beta_label",
    "sample_fraction",
    "samples_per_client",
] + [key for _, key in selected] + [key for _, key in pooled_selected]

with eval_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=csv_columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

eval_report = Path(os.environ["EVAL_REPORT"])
lines = []
lines.append("FedMIA BINN EVAL Report")
lines.append(f"experiment_id: {os.environ['EVAL_ID']}")
lines.append("script: membership_attack_binn_eval.sh")
lines.append(f"runs_per_config: {os.environ['RUNS_PER_CONFIG']}")
lines.append(f"config_reports_aggregated: {len(csv_paths)}")
lines.append(f"config_rows_aggregated: {len(rows)}")
lines.append(f"csv_file: {eval_csv}")
lines.append("")
lines.append("Source Config Reports")
for path in csv_paths:
    lines.append(f"  {path}")

lines.append("")
lines.append("Aggregate Across Launched Config Runs")
for label, key in selected:
    values = [as_float(row, key) for row in rows]
    lines.append(f"  {label}: {fmt(mean(values))} +/- {fmt(sample_std(values))}")

lines.append("")
lines.append("Aggregate Pooled Readouts Across Launched Config Runs")
for label, key in pooled_selected:
    values = [as_float(row, key) for row in rows]
    lines.append(f"  {label}: {fmt(mean(values))} +/- {fmt(sample_std(values))}")

lines.append("")
lines.append("Per-Config Results")
for index, row in enumerate(rows, start=1):
    descriptor = (
        f"config={index:03d} clients={row.get('clients')} rounds={row.get('rounds')} "
        f"local_epochs={row.get('local_epochs')} beta={row.get('beta_label')} "
        f"samples_per_client={row.get('samples_per_client')}"
    )
    lines.append(f"  {descriptor}")
    lines.append(
        "    utility "
        f"train_acc={fmt(as_float(row, 'train_acc_final_mean'))} "
        f"holdout_acc={fmt(as_float(row, 'holdout_acc_final_mean'))}"
    )
    lines.append(
        "    FedMIA-I loss "
        f"auc={fmt(as_float(row, 'fedmia_i_loss_auc_mean'))} "
        f"f1={fmt(as_float(row, 'fedmia_i_loss_f1_mean'))} "
        f"tpr={fmt(as_float(row, 'fedmia_i_loss_tpr_mean'))} "
        f"tnr={fmt(as_float(row, 'fedmia_i_loss_tnr_mean'))} "
        f"fpr={fmt(as_float(row, 'fedmia_i_loss_fpr_mean'))} "
        f"tpr@fpr0.001={fmt(as_float(row, 'fedmia_i_loss_tpr_at_fpr_0.001_mean'))}"
    )
    lines.append(
        "    FedMIA-II cosine "
        f"auc={fmt(as_float(row, 'fedmia_ii_cosine_auc_mean'))} "
        f"f1={fmt(as_float(row, 'fedmia_ii_cosine_f1_mean'))} "
        f"tpr={fmt(as_float(row, 'fedmia_ii_cosine_tpr_mean'))} "
        f"tnr={fmt(as_float(row, 'fedmia_ii_cosine_tnr_mean'))} "
        f"fpr={fmt(as_float(row, 'fedmia_ii_cosine_fpr_mean'))} "
        f"tpr@fpr0.001={fmt(as_float(row, 'fedmia_ii_cosine_tpr_at_fpr_0.001_mean'))}"
    )

eval_report.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo
echo "EVAL report written to ${EVAL_REPORT}"
echo "EVAL csv written to ${EVAL_CSV}"
