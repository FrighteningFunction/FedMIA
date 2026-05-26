from __future__ import annotations

import csv
import glob
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports"
CHART_DIR = REPO_ROOT / "charts"


LOSS_COLOR = "#226f54"
COSINE_COLOR = "#9b2226"
UTILITY_COLOR = "#335c67"
GRID_COLOR = "#d7d7d7"
TEXT_COLOR = "#222222"


def as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def as_int(row: Dict[str, str], key: str) -> int:
    value = as_float(row, key)
    if math.isnan(value):
        return 0
    return int(value)


def read_csv_rows(pattern: str, family: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(glob.glob(str(REPORT_DIR / pattern))):
        name = os.path.basename(path)
        if "patient_" in name:
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["family"] = family
                row["source_csv"] = os.path.relpath(path, REPO_ROOT)
                rows.append(row)
    return rows


def dedupe_latest(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    by_config: Dict[tuple, Dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("clients"),
            row.get("rounds"),
            row.get("local_epochs"),
            row.get("beta_label"),
            row.get("samples_per_client"),
        )
        if key not in by_config or row["source_csv"] > by_config[key]["source_csv"]:
            by_config[key] = row
    return list(by_config.values())


def beta_sort_value(beta_label: str) -> float:
    if beta_label == "iid":
        return 1000.0
    try:
        return float(beta_label)
    except ValueError:
        return -1.0


def sort_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            as_int(row, "clients"),
            as_int(row, "rounds"),
            as_int(row, "local_epochs"),
            beta_sort_value(row.get("beta_label", "")),
            as_int(row, "samples_per_client"),
            row.get("source_csv", ""),
        ),
    )


def load_binn_rows() -> List[Dict[str, str]]:
    rows = read_csv_rows("fedmia_binn_paper_grid_*.csv", "BINN")
    real_rows = [
        row
        for row in rows
        if as_int(row, "rounds") >= 10 and as_int(row, "samples_per_client") >= 30
        and as_float(row, "member_count_mean") >= 30
        and as_float(row, "nonmember_count_mean") >= 30
    ]
    return sort_rows(dedupe_latest(real_rows))


def load_cifar_rows() -> List[Dict[str, str]]:
    rows = read_csv_rows("fedmia_cifar100_alexnet_*.csv", "AlexNet/CIFAR100")
    real_rows = [
        row
        for row in rows
        if as_int(row, "rounds") >= 10 and as_int(row, "samples_per_client") >= 1000
    ]
    return sort_rows(dedupe_latest(real_rows))


def label_for(row: Dict[str, str]) -> str:
    return (
        f"{as_int(row, 'clients')}c/{as_int(row, 'rounds')}r\n"
        f"{as_int(row, 'local_epochs')}e beta={row.get('beta_label', '')}"
    )


def configure_axis(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=13, color=TEXT_COLOR, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_figure(fig, stem: str):
    CHART_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / f"{stem}.png", dpi=180)
    fig.savefig(CHART_DIR / f"{stem}.svg")
    plt.close(fig)


def grouped_bar_chart(
    rows: Sequence[Dict[str, str]],
    metric_suffix: str,
    title: str,
    ylabel: str,
    stem: str,
):
    if not rows:
        return
    labels = [label_for(row) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    loss_values = [as_float(row, f"fedmia_i_loss_{metric_suffix}_mean") for row in rows]
    cosine_values = [as_float(row, f"fedmia_ii_cosine_{metric_suffix}_mean") for row in rows]
    loss_std = [as_float(row, f"fedmia_i_loss_{metric_suffix}_std") for row in rows]
    cosine_std = [as_float(row, f"fedmia_ii_cosine_{metric_suffix}_std") for row in rows]

    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.35), 5.8))
    configure_axis(ax, title, ylabel)
    ax.bar(
        x - width / 2,
        loss_values,
        width,
        yerr=[0 if math.isnan(value) else value for value in loss_std],
        color=LOSS_COLOR,
        label="FedMIA-I loss",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        cosine_values,
        width,
        yerr=[0 if math.isnan(value) else value for value in cosine_std],
        color=COSINE_COLOR,
        label="FedMIA-II cosine",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend(frameon=False, loc="upper left", ncols=2)
    save_figure(fig, stem)


def utility_vs_attack(rows: Sequence[Dict[str, str]], utility_key: str, title: str, stem: str):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.set_title(title, fontsize=13, color=TEXT_COLOR, pad=14)
    ax.set_xlabel("Utility accuracy")
    ax.set_ylabel("Attack AUC")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for row in rows:
        utility = as_float(row, utility_key)
        loss_auc = as_float(row, "fedmia_i_loss_auc_mean")
        cosine_auc = as_float(row, "fedmia_ii_cosine_auc_mean")
        label = label_for(row).replace("\n", " ")
        ax.scatter(utility, loss_auc, color=LOSS_COLOR, s=55, alpha=0.9)
        ax.scatter(utility, cosine_auc, color=COSINE_COLOR, marker="s", s=45, alpha=0.9)
        ax.annotate(label, (utility, max(loss_auc, cosine_auc)), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.scatter([], [], color=LOSS_COLOR, label="FedMIA-I loss")
    ax.scatter([], [], color=COSINE_COLOR, marker="s", label="FedMIA-II cosine")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, stem)


def write_chart_data_csv(rows: Sequence[Dict[str, str]]):
    fields = [
        "family",
        "source_csv",
        "clients",
        "rounds",
        "local_epochs",
        "beta_label",
        "samples_per_client",
        "holdout_acc_final_mean",
        "test_acc_final_mean",
        "fedmia_i_loss_auc_mean",
        "fedmia_i_loss_f1_mean",
        "fedmia_i_loss_tpr_at_fpr_0.001_mean",
        "fedmia_ii_cosine_auc_mean",
        "fedmia_ii_cosine_f1_mean",
        "fedmia_ii_cosine_tpr_at_fpr_0.001_mean",
    ]
    CHART_DIR.mkdir(exist_ok=True)
    with open(CHART_DIR / "fedmia_chart_data.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(binn_rows: Sequence[Dict[str, str]], cifar_rows: Sequence[Dict[str, str]]):
    lines = [
        "# FedMIA Run Charts",
        "",
        "Generated from existing report CSV files in `reports/`.",
        "",
        "Main filters:",
        "- BINN paper-style charts include rows with `rounds >= 10` and `samples_per_client >= 30`.",
        "- BINN rows also require `member_count_mean >= 30` and `nonmember_count_mean >= 30`.",
        "- AlexNet/CIFAR100 charts include rows with `rounds >= 10` and `samples_per_client >= 1000`.",
        "- Plumbing/debug runs are excluded from the main figures.",
        "- Severely imbalanced Dirichlet rows where the target client collapsed to a tiny candidate set are excluded.",
        "- Duplicate configurations are deduplicated by keeping the latest CSV file.",
        "",
        "Charts:",
        "- `binn_attack_auc_by_config.png/svg`: FedMIA-I and FedMIA-II AUC by BINN configuration.",
        "- `binn_attack_f1_by_config.png/svg`: F1 at `delta=0.5` by BINN configuration.",
        "- `binn_low_fpr_tpr_by_config.png/svg`: TPR at FPR=0.001 by BINN configuration.",
        "- `binn_utility_vs_attack_auc.png/svg`: holdout accuracy versus attack AUC for BINN.",
        "- `cifar100_alexnet_attack_auc_by_config.png/svg`: FedMIA-I and FedMIA-II AUC by CIFAR100/AlexNet configuration.",
        "- `cifar100_alexnet_attack_f1_by_config.png/svg`: F1 at `delta=0.5` by CIFAR100/AlexNet configuration.",
        "- `cifar100_alexnet_utility_vs_attack_auc.png/svg`: test accuracy versus attack AUC for CIFAR100/AlexNet.",
        "- `fedmia_chart_data.csv`: source rows used for all charts.",
        "",
        "Included BINN rows:",
    ]
    for row in binn_rows:
        lines.append(
            f"- `{row['source_csv']}`: {label_for(row).replace(chr(10), ', ')} "
            f"loss_auc={as_float(row, 'fedmia_i_loss_auc_mean'):.3f}, "
            f"cos_auc={as_float(row, 'fedmia_ii_cosine_auc_mean'):.3f}, "
            f"holdout_acc={as_float(row, 'holdout_acc_final_mean'):.3f}"
        )
    lines.extend(["", "Included AlexNet/CIFAR100 rows:"])
    for row in cifar_rows:
        lines.append(
            f"- `{row['source_csv']}`: {label_for(row).replace(chr(10), ', ')} "
            f"loss_auc={as_float(row, 'fedmia_i_loss_auc_mean'):.3f}, "
            f"cos_auc={as_float(row, 'fedmia_ii_cosine_auc_mean'):.3f}, "
            f"test_acc={as_float(row, 'test_acc_final_mean'):.3f}"
        )
    with open(CHART_DIR / "README.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    CHART_DIR.mkdir(exist_ok=True)
    binn_rows = load_binn_rows()
    cifar_rows = load_cifar_rows()

    grouped_bar_chart(
        binn_rows,
        "auc",
        "BINN FedMIA Attack AUC By Configuration",
        "AUC",
        "binn_attack_auc_by_config",
    )
    grouped_bar_chart(
        binn_rows,
        "f1",
        "BINN FedMIA F1 By Configuration",
        "F1 at delta=0.5",
        "binn_attack_f1_by_config",
    )
    grouped_bar_chart(
        binn_rows,
        "tpr_at_fpr_0.001",
        "BINN FedMIA TPR At FPR=0.001",
        "TPR at FPR=0.001",
        "binn_low_fpr_tpr_by_config",
    )
    utility_vs_attack(
        binn_rows,
        "holdout_acc_final_mean",
        "BINN Utility Versus Attack AUC",
        "binn_utility_vs_attack_auc",
    )

    grouped_bar_chart(
        cifar_rows,
        "auc",
        "AlexNet/CIFAR100 FedMIA Attack AUC By Configuration",
        "AUC",
        "cifar100_alexnet_attack_auc_by_config",
    )
    grouped_bar_chart(
        cifar_rows,
        "f1",
        "AlexNet/CIFAR100 FedMIA F1 By Configuration",
        "F1 at delta=0.5",
        "cifar100_alexnet_attack_f1_by_config",
    )
    utility_vs_attack(
        cifar_rows,
        "test_acc_final_mean",
        "AlexNet/CIFAR100 Utility Versus Attack AUC",
        "cifar100_alexnet_utility_vs_attack_auc",
    )

    all_rows = [{**row} for row in binn_rows] + [{**row} for row in cifar_rows]
    write_chart_data_csv(all_rows)
    write_readme(binn_rows, cifar_rows)
    print(f"wrote {len(list(CHART_DIR.glob('*')))} files to {CHART_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
