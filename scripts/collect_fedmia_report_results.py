"""Collect FedMIA experiment reports into reusable aggregate CSV tables.

The script scans the active ``reports/`` directory, parses BINN and
CIFAR100/AlexNet report text files, extracts configuration and metric rows
including pooled results and patient-vulnerability summaries, and writes the
cleaned CSV files used by the chart and table-view scripts.
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "aggregated_report"

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan"
PAIR_RE = re.compile(
    rf"(?P<key>[A-Za-z0-9_.@]+)=(?P<value>{NUMBER_RE})(?:\s*\+/-\s*(?P<std>{NUMBER_RE}))?"
)
CONFIG_RE = re.compile(
    r"config=(?P<config_id>\d+)\s+clients=(?P<clients>\d+)\s+rounds=(?P<rounds>\d+)\s+"
    r"local_epochs=(?P<local_epochs>\d+)\s+beta=(?P<beta_label>\S+)\s+"
    r"samples_per_client=(?P<samples_per_client>\d+)"
)
METHOD_RE = re.compile(r"(?P<label>FedMIA-[IVX]+)\s+(?P<channel>\w+)")


OUTPUT_COLUMNS = [
    "dataset_model",
    "source_file",
    "experiment_id",
    "script",
    "config_label",
    "aggregation",
    "measurement",
    "method",
    "clients",
    "rounds",
    "local_epochs",
    "beta_label",
    "samples_per_client",
    "nonmember_source",
    "runs",
    "runs_per_config",
    "candidate_count",
    "audit_patient_count",
    "holdout_fraction",
    "threshold_delta",
    "member_count",
    "nonmember_count",
    "train_acc",
    "holdout_acc",
    "test_acc",
    "auc",
    "f1",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "precision",
    "recall",
    "threshold_accuracy",
    "log_auc",
    "tpr_at_fpr_0.001",
    "tpr_at_fpr_0.01",
    "tpr_at_fpr_0.1",
    "best_f1",
    "best_threshold",
    "fedmia_i_loss_patient_eligible",
    "fedmia_i_loss_patient_auc",
    "fedmia_i_loss_patient_score_gap",
    "fedmia_i_loss_most_vulnerable_patient",
    "fedmia_i_loss_most_vulnerable_auc",
    "fedmia_i_loss_most_vulnerable_score_gap",
    "fedmia_i_loss_least_vulnerable_patient",
    "fedmia_i_loss_least_vulnerable_auc",
    "fedmia_i_loss_least_vulnerable_score_gap",
    "fedmia_ii_cosine_patient_eligible",
    "fedmia_ii_cosine_patient_auc",
    "fedmia_ii_cosine_patient_score_gap",
    "fedmia_ii_cosine_most_vulnerable_patient",
    "fedmia_ii_cosine_most_vulnerable_auc",
    "fedmia_ii_cosine_most_vulnerable_score_gap",
    "fedmia_ii_cosine_least_vulnerable_patient",
    "fedmia_ii_cosine_least_vulnerable_auc",
    "fedmia_ii_cosine_least_vulnerable_score_gap",
]

RESULT_METRICS = [
    "auc",
    "f1",
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "precision",
    "recall",
    "threshold_accuracy",
    "log_auc",
    "tpr_at_fpr_0.001",
    "tpr_at_fpr_0.01",
    "tpr_at_fpr_0.1",
    "best_f1",
    "best_threshold",
]

UTILITY_SOURCES = {
    "train_acc": "train_acc_final",
    "holdout_acc": "holdout_acc_final",
    "test_acc": "test_acc_final",
}

PATIENT_SUMMARY_SOURCES = {
    "fedmia_i_loss_patient_eligible": "fedmia_i_loss_patient_eligible_patients",
    "fedmia_i_loss_patient_auc": "fedmia_i_loss_patient_auc_mean",
    "fedmia_i_loss_patient_score_gap": "fedmia_i_loss_patient_score_gap_mean",
    "fedmia_i_loss_most_vulnerable_patient": "fedmia_i_loss_patient_most_vulnerable_patient",
    "fedmia_i_loss_most_vulnerable_auc": "fedmia_i_loss_patient_most_vulnerable_auc",
    "fedmia_i_loss_most_vulnerable_score_gap": "fedmia_i_loss_patient_most_vulnerable_score_gap",
    "fedmia_i_loss_least_vulnerable_patient": "fedmia_i_loss_patient_least_vulnerable_patient",
    "fedmia_i_loss_least_vulnerable_auc": "fedmia_i_loss_patient_least_vulnerable_auc",
    "fedmia_i_loss_least_vulnerable_score_gap": "fedmia_i_loss_patient_least_vulnerable_score_gap",
    "fedmia_ii_cosine_patient_eligible": "fedmia_ii_cosine_patient_eligible_patients",
    "fedmia_ii_cosine_patient_auc": "fedmia_ii_cosine_patient_auc_mean",
    "fedmia_ii_cosine_patient_score_gap": "fedmia_ii_cosine_patient_score_gap_mean",
    "fedmia_ii_cosine_most_vulnerable_patient": "fedmia_ii_cosine_patient_most_vulnerable_patient",
    "fedmia_ii_cosine_most_vulnerable_auc": "fedmia_ii_cosine_patient_most_vulnerable_auc",
    "fedmia_ii_cosine_most_vulnerable_score_gap": "fedmia_ii_cosine_patient_most_vulnerable_score_gap",
    "fedmia_ii_cosine_least_vulnerable_patient": "fedmia_ii_cosine_patient_least_vulnerable_patient",
    "fedmia_ii_cosine_least_vulnerable_auc": "fedmia_ii_cosine_patient_least_vulnerable_auc",
    "fedmia_ii_cosine_least_vulnerable_score_gap": "fedmia_ii_cosine_patient_least_vulnerable_score_gap",
}


def normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace("@", "_at_")
    key = key.replace("-", "_")
    key = key.replace(" ", "_")
    key = key.replace("train_acc_final", "train_acc")
    key = key.replace("holdout_acc_final", "holdout_acc")
    key = key.replace("test_acc_final", "test_acc")
    key = key.replace("threshold_delta", "threshold_delta")
    return key


def parse_scalar(value: str):
    value = str(value).strip()
    if value == "":
        return ""
    if value.lower() == "nan":
        return "nan"
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_pairs(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for match in PAIR_RE.finditer(text):
        key = normalize_key(match.group("key"))
        out[key] = parse_scalar(match.group("value"))
        if match.group("std") is not None:
            out[f"{key}_std"] = parse_scalar(match.group("std"))
    return out


def report_family(path: Path) -> Optional[str]:
    name = path.name
    if "_patient_" in name or name.endswith("_patient_metrics.txt"):
        return None
    if "fedmia_binn_lira_protocol" in name:
        return None
    if name.startswith("fedmia_binn_paper_grid_"):
        return "BINN"
    if name.startswith("fedmia_cifar100_alexnet_"):
        return "AlexNet/CIFAR100"
    return None


def parse_header(lines: Sequence[str], path: Path, family: str) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "dataset_model": family,
        "source_file": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "report_title": lines[0].strip() if lines else "",
    }
    for line in lines[:30]:
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = normalize_key(key)
        if key in {"experiment_id", "script"}:
            metadata[key] = value.strip()

    in_config = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Configuration":
            in_config = True
            continue
        if in_config and stripped and not line.startswith(" "):
            break
        if in_config and ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[normalize_key(key)] = parse_scalar(value)
    return metadata


def method_name(label: str, channel: str) -> Tuple[str, str]:
    channel = channel.strip().lower()
    if label.lower() == "fedmia-i":
        return f"{label} {channel}", f"fedmia_i_{channel}"
    if label.lower() == "fedmia-ii":
        return f"{label} {channel}", f"fedmia_ii_{channel}"
    return f"{label} {channel}", channel


def config_label(row: Dict[str, object]) -> str:
    return (
        f"{row.get('clients')}c/{row.get('rounds')}r "
        f"{row.get('local_epochs')}e beta={row.get('beta_label')} "
        f"spc={row.get('samples_per_client')}"
    )


def load_companion_csv(path: Path) -> Dict[str, Dict[str, str]]:
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row.get("config_id", ""): row for row in csv.DictReader(handle)}


def csv_metric_name(metric: str) -> str:
    if metric == "best_threshold":
        return "best_f1_threshold"
    return metric


def set_from_source(
    row: Dict[str, object],
    source: Dict[str, str],
    source_key: str,
    target_key: str,
):
    if source.get(source_key, "") != "":
        row[target_key] = parse_scalar(source[source_key])


def enrich_from_companion(row: Dict[str, object], companion: Dict[str, Dict[str, str]]):
    source = companion.get(str(row.get("config_id", "")), {})
    if not source:
        return

    for target_key, source_prefix in UTILITY_SOURCES.items():
        set_from_source(row, source, f"{source_prefix}_mean", target_key)
        set_from_source(row, source, f"{source_prefix}_std", f"{target_key}_std")

    set_from_source(row, source, "member_count_mean", "member_count")
    set_from_source(row, source, "nonmember_count_mean", "nonmember_count")
    set_from_source(row, source, "nonmember_source", "nonmember_source")
    set_from_source(row, source, "audit_patient_count", "audit_patient_count")

    for target_key, source_key in PATIENT_SUMMARY_SOURCES.items():
        set_from_source(row, source, source_key, target_key)
    for target_key, source_key in (
        ("fedmia_i_loss_patient_auc", "fedmia_i_loss_patient_auc_std"),
        ("fedmia_i_loss_patient_score_gap", "fedmia_i_loss_patient_score_gap_std"),
        ("fedmia_ii_cosine_patient_auc", "fedmia_ii_cosine_patient_auc_std"),
        ("fedmia_ii_cosine_patient_score_gap", "fedmia_ii_cosine_patient_score_gap_std"),
    ):
        set_from_source(row, source, source_key, f"{target_key}_std")

    measurement = row.get("measurement")
    aggregation = row.get("aggregation")
    if not measurement or aggregation not in {"mean", "pooled"}:
        return

    for metric in RESULT_METRICS:
        source_metric = csv_metric_name(metric)
        if aggregation == "mean":
            set_from_source(row, source, f"{measurement}_{source_metric}_mean", metric)
            set_from_source(row, source, f"{measurement}_{source_metric}_std", f"{metric}_std")
        else:
            set_from_source(row, source, f"{measurement}_pooled_{source_metric}", metric)

    if aggregation == "pooled":
        set_from_source(row, source, f"{measurement}_pooled_member_count", "member_count")
        set_from_source(row, source, f"{measurement}_pooled_nonmember_count", "nonmember_count")


def parse_grid_report(path: Path, family: str) -> List[Dict[str, object]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = parse_header(lines, path, family)
    companion = load_companion_csv(path)
    rows: List[Dict[str, object]] = []
    in_grid = False
    current_config: Dict[str, object] = {}
    utility: Dict[str, object] = {}

    for line in lines:
        stripped = line.strip()
        if stripped == "Grid Results":
            in_grid = True
            continue
        if in_grid and stripped and not line.startswith(" "):
            break
        if not in_grid or not stripped:
            continue

        config_match = CONFIG_RE.search(stripped)
        if config_match:
            current_config = {
                key: parse_scalar(value) for key, value in config_match.groupdict().items()
            }
            utility = {}
            continue

        if stripped.startswith("utility "):
            utility.update(parse_pairs(stripped))
            continue
        if stripped.startswith("train_acc_final=") or stripped.startswith("holdout_acc_final="):
            utility.update(parse_pairs(stripped))
            continue

        if not stripped.startswith("FedMIA-"):
            continue
        method_match = METHOD_RE.search(stripped)
        if not method_match:
            continue
        label = method_match.group("label")
        channel = method_match.group("channel")
        aggregation = "pooled" if " pooled " in f" {stripped} " else "mean"
        if aggregation == "pooled" and channel == "pooled":
            if label.lower() == "fedmia-i":
                channel = "loss"
            elif label.lower() == "fedmia-ii":
                channel = "cosine"
            metric_text = stripped.split("pooled", 1)[1]
        else:
            metric_text = stripped.split(channel, 1)[1]
        method, measurement = method_name(label, channel)

        row: Dict[str, object] = {}
        row.update(metadata)
        row.update(current_config)
        row.update(utility)
        row["config_label"] = config_label(row)
        row["aggregation"] = aggregation
        row["method"] = method
        row["measurement"] = measurement
        row.update(parse_pairs(metric_text))
        enrich_from_companion(row, companion)
        rows.append(row)
    return rows


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def keep_row(row: Dict[str, object]) -> bool:
    rounds = to_float(row.get("rounds"))
    samples_per_client = to_float(row.get("samples_per_client"))
    if rounds < 10:
        return False
    if row.get("dataset_model") == "BINN":
        member_count = to_float(row.get("member_count"))
        nonmember_count = to_float(row.get("nonmember_count"))
        return samples_per_client >= 30 and member_count >= 30 and nonmember_count >= 30
    if row.get("dataset_model") == "AlexNet/CIFAR100":
        return samples_per_client >= 1000
    return False


def collect(report_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    binn_rows: List[Dict[str, object]] = []
    cifar_rows: List[Dict[str, object]] = []
    for path in sorted(report_dir.rglob("*.txt") if report_dir.exists() else []):
        family = report_family(path)
        if family is None:
            continue
        rows = [row for row in parse_grid_report(path, family) if keep_row(row)]
        if family == "BINN":
            binn_rows.extend(rows)
        elif family == "AlexNet/CIFAR100":
            cifar_rows.extend(rows)
    return binn_rows, cifar_rows


def deduplicate_latest(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        key = (
            row.get("dataset_model"),
            row.get("config_label"),
            row.get("nonmember_source"),
            row.get("aggregation"),
            row.get("measurement"),
        )
        current = by_key.get(key)
        if current is None or str(row.get("source_file", "")) > str(current.get("source_file", "")):
            by_key[key] = row
    return sorted(
        by_key.values(),
        key=lambda row: (
            str(row.get("dataset_model", "")),
            int(to_float(row.get("clients"))),
            int(to_float(row.get("rounds"))),
            int(to_float(row.get("local_epochs"))),
            str(row.get("beta_label", "")),
            int(to_float(row.get("samples_per_client"))),
            str(row.get("nonmember_source", "")),
            str(row.get("measurement", "")),
            str(row.get("aggregation", "")),
        ),
    )


def format_number(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def has_nonzero_std(value) -> bool:
    if value is None or value == "" or value == "nan":
        return False
    try:
        return abs(float(value)) > 1e-12
    except (TypeError, ValueError):
        return False


def presentation_row(row: Dict[str, object]) -> Dict[str, str]:
    formatted: Dict[str, str] = {}
    for column in OUTPUT_COLUMNS:
        value = row.get(column, "")
        std_value = row.get(f"{column}_std", "")
        if has_nonzero_std(std_value):
            formatted[column] = f"{format_number(value)} +/- {format_number(std_value)}"
        else:
            formatted[column] = format_number(value)
    return formatted


def write_csv(path: Path, rows: Sequence[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(presentation_row(row) for row in rows)


def write_readme(path: Path, binn_rows: Sequence[Dict[str, object]], cifar_rows: Sequence[Dict[str, object]]):
    lines = [
        "# Aggregated FedMIA Report Tables",
        "",
        "Generated by `scripts/collect_fedmia_report_results.py`.",
        "",
        "Included report families:",
        "- `fedmia_binn_paper_grid_*.txt`",
        "- `fedmia_cifar100_alexnet_*.txt`",
        "",
        "Excluded:",
        "- patient-level reports",
        "- BINN LiRA-style protocol reports",
        "- early BINN research reports",
        "- plumbing/debug runs",
        "- BINN rows with fewer than 30 member or nonmember candidates",
        "- older duplicate rows when the same configuration/method/aggregation was rerun",
        "- nonmember-source modes are deduplicated separately",
        "- separate `*_std` columns; nonzero standard deviations are folded into `value +/- std` cells",
        "",
        "Output files:",
        "- `binn_fedmia_results.csv`",
        "- `cifar100_alexnet_fedmia_results.csv`",
        "- `fedmia_paper_table_view.html` (generated by `scripts/make_fedmia_table_view.py`)",
        "- `fedmia_paper_table_view.csv`",
        "- `fedmia_paper_table_view.tsv`",
        "",
        "New BINN experiment folders are scanned recursively under `reports/`.",
        "Patient-vulnerability columns are populated when the source experiment CSV",
        "contains per-patient summaries.",
        "",
        "Open `fedmia_paper_table_view.html` in a browser to filter rows, choose columns,",
        "copy TSV tables for Word/LaTeX helpers, or download selected rows as a smaller CSV.",
        "",
        f"BINN rows written after deduplication: {len(binn_rows)}",
        f"CIFAR100/AlexNet rows written after deduplication: {len(cifar_rows)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        report_dir = root / "reports"
        output_dir = root / "aggregated_report"
        report_dir.mkdir()

        valid_binn = report_dir / "fedmia_binn_paper_grid_fake.txt"
        valid_binn.write_text(
            """FedMIA BINN Paper-Style Grid Report
experiment_id: fake_binn
script: experiments/fake.py

Configuration
  runs_per_config: 2
  candidate_count: 0
  threshold_delta: 0.5

Grid Results
  config=001 clients=10 rounds=100 local_epochs=2 beta=iid samples_per_client=91
    train_acc_final=0.900000 +/- 0.100000
    holdout_acc_final=0.800000 +/- 0.050000
    FedMIA-I loss auc=0.600000 +/- 0.010000 f1=0.700000 +/- 0.020000 tpr@fpr0.001=0.030000
    FedMIA-I pooled auc=0.610000 f1=0.710000 tpr=0.800000 tnr=0.400000 fpr=0.600000 tpr@fpr0.001=0.040000
""",
            encoding="utf-8",
        )
        valid_binn.with_suffix(".csv").write_text(
            (
                "config_id,member_count_mean,nonmember_count_mean,"
                "fedmia_i_loss_tpr_mean,fedmia_i_loss_tpr_std,"
                "fedmia_i_loss_tnr_mean,fedmia_i_loss_tnr_std,"
                "fedmia_i_loss_fpr_mean,fedmia_i_loss_fpr_std,"
                "fedmia_i_loss_pooled_tpr,fedmia_i_loss_pooled_tnr,"
                "fedmia_i_loss_pooled_fpr\n"
                "1,91,91,0.75,0.05,0.55,0.04,0.45,0.04,0.80,0.40,0.60\n"
            ),
            encoding="utf-8",
        )

        bad_binn = report_dir / "fedmia_binn_paper_grid_bad.txt"
        bad_binn.write_text(
            """FedMIA BINN Paper-Style Grid Report
experiment_id: bad_binn
script: experiments/fake.py

Grid Results
  config=001 clients=5 rounds=100 local_epochs=2 beta=0.1 samples_per_client=182
    train_acc_final=0.700000 +/- 0.000000
    holdout_acc_final=0.600000 +/- 0.000000
    FedMIA-I loss auc=1.000000 +/- 0.000000 f1=0.660000 +/- 0.000000 tpr@fpr0.001=1.000000
""",
            encoding="utf-8",
        )
        bad_binn.with_suffix(".csv").write_text(
            "config_id,member_count_mean,nonmember_count_mean\n1,1,1\n",
            encoding="utf-8",
        )

        cifar = report_dir / "fedmia_cifar100_alexnet_fake.txt"
        cifar.write_text(
            """FedMIA CIFAR100 AlexNet Grid Report
experiment_id: fake_cifar
script: experiments/fake_cifar.py

Configuration
  runs_per_config: 5
  candidate_count: 512
  threshold_delta: 0.5

Grid Results
  config=001 clients=10 rounds=100 local_epochs=5 beta=0.1 samples_per_client=5000
    utility train_acc=0.460000 +/- 0.010000 test_acc=0.210000 +/- 0.020000
    FedMIA-II cosine auc=0.950000 +/- 0.030000 f1=0.500000 +/- 0.040000 tpr@fpr0.001=0.400000
""",
            encoding="utf-8",
        )

        lira = report_dir / "fedmia_binn_lira_protocol_fake.txt"
        lira.write_text("FedMIA BINN LiRA-Style Patient Protocol Report\n", encoding="utf-8")
        patient = report_dir / "fedmia_binn_paper_grid_fake_patient_metrics.txt"
        patient.write_text("FedMIA BINN Patient-Level Metrics Report\n", encoding="utf-8")

        binn_rows, cifar_rows = collect(report_dir)
        binn_rows = deduplicate_latest(binn_rows)
        cifar_rows = deduplicate_latest(cifar_rows)
        assert len(binn_rows) == 2, f"expected 2 BINN rows, got {len(binn_rows)}"
        assert len(cifar_rows) == 1, f"expected 1 CIFAR row, got {len(cifar_rows)}"
        assert all(row["experiment_id"] != "bad_binn" for row in binn_rows), "bad tiny-candidate BINN row leaked in"
        assert binn_rows[0]["auc"] == 0.6, "BINN mean AUC parse failed"
        assert binn_rows[0]["tpr"] == 0.75 and binn_rows[0]["tnr"] == 0.55, "BINN CSV metric enrichment failed"
        assert binn_rows[1]["aggregation"] == "pooled" and binn_rows[1]["auc"] == 0.61, "BINN pooled row parse failed"
        assert binn_rows[1]["tpr"] == 0.80 and binn_rows[1]["tnr"] == 0.40, "BINN pooled CSV metric enrichment failed"
        assert binn_rows[1]["measurement"] == "fedmia_i_loss", "BINN pooled measurement label failed"
        assert cifar_rows[0]["measurement"] == "fedmia_ii_cosine", "CIFAR measurement parse failed"

        write_csv(output_dir / "binn_fedmia_results.csv", binn_rows)
        write_csv(output_dir / "cifar100_alexnet_fedmia_results.csv", cifar_rows)
        assert (output_dir / "binn_fedmia_results.csv").exists()
        assert (output_dir / "cifar100_alexnet_fedmia_results.csv").exists()
        lines = (output_dir / "binn_fedmia_results.csv").read_text(encoding="utf-8").splitlines()
        header = lines[0]
        assert "round_grid" not in header and "client_grid" not in header, "grid metadata leaked into output"
        assert "auc_std" not in header and "f1_std" not in header, "std columns leaked into output"
        assert "0.600000 +/- 0.010000" in lines[1], "value/std presentation formatting failed"
        print("self-test passed")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate FedMIA report results into CSV tables.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    binn_rows, cifar_rows = collect(args.report_dir)
    binn_rows = deduplicate_latest(binn_rows)
    cifar_rows = deduplicate_latest(cifar_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "binn_fedmia_results.csv", binn_rows)
    write_csv(args.output_dir / "cifar100_alexnet_fedmia_results.csv", cifar_rows)
    write_readme(args.output_dir / "README.md", binn_rows, cifar_rows)
    print(f"wrote {len(binn_rows)} BINN rows and {len(cifar_rows)} CIFAR100/AlexNet rows to {args.output_dir}")


if __name__ == "__main__":
    main()
