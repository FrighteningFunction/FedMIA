from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import FedMIAConfig, score_rounds  # noqa: E402
from models.binn import binn  # noqa: E402
from utils.federated import fed_avg_state_dicts  # noqa: E402

import experiments.fedmia_binn_research as research  # noqa: E402


LOGGER = logging.getLogger("fedmia_binn_lira_protocol")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
PRIMARY_MEASUREMENTS = ("cosine", "loss", "combined")
WINDOWS = ("all", "last_half", "last_quarter")
MEASUREMENTS = PRIMARY_MEASUREMENTS + tuple(
    f"{measurement}_{window}"
    for window in WINDOWS
    if window != "all"
    for measurement in PRIMARY_MEASUREMENTS
)


@dataclass
class RunSummary:
    run_id: int
    seed: int
    initial_acc: float
    final_acc: float
    included_audited_patients: int
    elapsed_seconds: float
    metrics: Dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser(
        description="LiRA-style repeated-patient FedMIA evaluation on federated BINN."
    )
    parser.add_argument("--runs", type=int, default=200, help="independent FL trajectories")
    parser.add_argument("--rounds", type=int, default=20, help="communication rounds per trajectory")
    parser.add_argument("--local-epochs", type=int, default=2, help="local epochs per round")
    parser.add_argument("--num-clients", type=int, default=5, help="federated clients per trajectory")
    parser.add_argument("--samples-per-client", type=int, default=64, help="background samples per client")
    parser.add_argument("--audit-count", type=int, default=64, help="patients audited across trajectories")
    parser.add_argument("--inclusion-prob", type=float, default=0.5, help="P(patient IN target client)")
    parser.add_argument("--batch-size", type=int, default=16, help="local training batch size")
    parser.add_argument("--lr", type=float, default=0.03, help="local optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="local optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--threshold", type=float, default=0.5, help="reported FedMIA delta threshold")
    parser.add_argument(
        "--threshold-grid",
        default="0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="comma-separated deltas used for best-F1 diagnostics",
    )
    parser.add_argument("--outlier-std-factor", type=float, default=3.0, help="FedMIA sigma filter")
    parser.add_argument("--min-variance", type=float, default=1e-8, help="FedMIA Gaussian variance floor")
    parser.add_argument("--seed", type=int, default=20260509, help="base random seed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="training device")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--x-file", default="pnet_x.npy", help="ProstateCancer feature array")
    parser.add_argument("--y-file", default="pnet_y.npy", help="ProstateCancer label array")
    parser.add_argument("--max-samples", type=int, default=0, help="debug sample cap; 0 uses all")
    parser.add_argument("--feature-limit", type=int, default=0, help="debug feature cap; 0 uses all")
    parser.add_argument("--report-dir", default=REPORT_DIR, help="directory for final reports")
    parser.add_argument("--log-dir", default=LOG_DIR, help="directory for execution logs")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def make_experiment_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"fedmia_binn_lira_protocol_{stamp}_{uuid.uuid4().hex[:8]}"


def configure_logging(args, experiment_id: str) -> Tuple[str, str]:
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{experiment_id}.log")
    jsonl_path = os.path.join(args.log_dir, f"{experiment_id}.jsonl")
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    research.LOGGER = LOGGER
    return log_path, jsonl_path


def append_jsonl(path: str, event: Dict):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args):
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def balanced_sample_from_pool(labels, count: int, rng, pool, exclude=None) -> List[int]:
    if count <= 0:
        return []
    exclude = set(exclude or [])
    pool = [idx for idx in pool if idx not in exclude]
    label_values = labels.view(-1).cpu().numpy()
    selected = []
    per_class = count // 2
    for label in (0.0, 1.0):
        candidates = [idx for idx in pool if label_values[idx] == label and idx not in selected]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    if len(selected) < count:
        remaining = [idx for idx in pool if idx not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"Not enough samples to draw {count} records from the available pool.")
    rng.shuffle(selected)
    return selected[:count]


def make_membership_matrix(runs: int, audit_count: int, inclusion_prob: float, rng) -> np.ndarray:
    matrix = rng.random((runs, audit_count)) < inclusion_prob
    if runs > 1:
        for patient_col in range(audit_count):
            if matrix[:, patient_col].all():
                matrix[rng.integers(0, runs), patient_col] = False
            elif not matrix[:, patient_col].any():
                matrix[rng.integers(0, runs), patient_col] = True
    for run_idx in range(runs):
        if matrix[run_idx].all():
            matrix[run_idx, rng.integers(0, audit_count)] = False
        elif not matrix[run_idx].any():
            matrix[run_idx, rng.integers(0, audit_count)] = True
    return matrix


def make_client_indices_for_run(audit_indices, membership_flags, labels, args, rng):
    audit_set = set(audit_indices)
    all_indices = list(range(labels.shape[0]))
    background_pool = [idx for idx in all_indices if idx not in audit_set]
    included = [idx for idx, is_in in zip(audit_indices, membership_flags) if is_in]

    used = set(included)
    target_fill_count = max(args.samples_per_client - len(included), 0)
    target_fill = balanced_sample_from_pool(labels, target_fill_count, rng, background_pool, used)
    target_indices = included + target_fill
    used.update(target_fill)

    client_indices = [target_indices]
    for _ in range(1, args.num_clients):
        client_sample = balanced_sample_from_pool(labels, args.samples_per_client, rng, background_pool, used)
        client_indices.append(client_sample)
        used.update(client_sample)
    return client_indices


def make_loaders(x, y, client_indices, args, seed):
    generator = torch.Generator().manual_seed(seed)
    client_datasets = [TensorDataset(x[indices], y[indices]) for indices in client_indices]
    client_loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
        )
        for dataset in client_datasets
    ]
    combined_dataset = TensorDataset(
        torch.cat([dataset.tensors[0] for dataset in client_datasets], dim=0),
        torch.cat([dataset.tensors[1] for dataset in client_datasets], dim=0),
    )
    combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=False)
    return client_datasets, client_loaders, combined_loader


def split_scores_by_label(scores, membership_flags):
    member_scores = [score for score, label in zip(scores, membership_flags) if label]
    nonmember_scores = [score for score, label in zip(scores, membership_flags) if not label]
    return member_scores, nonmember_scores


def summarize_scores(scores_by_measurement, membership_flags, args):
    metrics = {}
    threshold_grid = research.parse_threshold_grid(args.threshold_grid)
    for measurement_name, scores in scores_by_measurement.items():
        member_scores, nonmember_scores = split_scores_by_label(scores, membership_flags)
        measurement_metrics, _ = research.score_list_metrics(
            member_scores,
            nonmember_scores,
            args.threshold,
            threshold_grid,
        )
        metrics.update(research.prefix_metrics(measurement_name, measurement_metrics))
    return metrics


def window_rounds(rounds, window: str):
    if window == "all":
        return rounds
    if not rounds:
        return rounds
    if window == "last_half":
        start = len(rounds) // 2
    elif window == "last_quarter":
        keep = max(1, math.ceil(len(rounds) * 0.25))
        start = len(rounds) - keep
    else:
        raise ValueError(f"Unknown scoring window: {window}")
    return rounds[start:]


def windowed_scores(cosine_rounds, loss_rounds, fedmia_config):
    """
    Score all rounds and late-round windows.

    FedMIA averages evidence across communication rounds, but early rounds can
    be noisy. Keeping late-window diagnostics lets us see whether membership
    signal strengthens after the model has learned more.
    """
    scores_by_measurement = {}
    for window in WINDOWS:
        suffix = "" if window == "all" else f"_{window}"
        cosine_scores = score_rounds(window_rounds(cosine_rounds, window), fedmia_config).aggregate_scores
        loss_scores = score_rounds(window_rounds(loss_rounds, window), fedmia_config).aggregate_scores
        combined_scores = [
            (cosine_score + loss_score) / 2.0
            for cosine_score, loss_score in zip(cosine_scores, loss_scores)
        ]
        scores_by_measurement[f"cosine{suffix}"] = cosine_scores
        scores_by_measurement[f"loss{suffix}"] = loss_scores
        scores_by_measurement[f"combined{suffix}"] = combined_scores
    return scores_by_measurement


def run_one_trajectory(
    run_id,
    run_seed,
    graph,
    x,
    y,
    audit_indices,
    membership_flags,
    args,
    device,
    jsonl_path,
):
    run_start = time.time()
    set_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    client_indices = make_client_indices_for_run(audit_indices, membership_flags, y, args, rng)
    client_datasets, client_loaders, combined_loader = make_loaders(x, y, client_indices, args, run_seed)
    audited_samples = [(x[idx], y[idx]) for idx in audit_indices]

    global_model = binn(graph=graph, output_size=1, dropout_prob=0.0, output_last_layers=1).to(device)
    initial_acc = research.model_accuracy(global_model, combined_loader, device)
    cosine_rounds = []
    loss_rounds = []

    LOGGER.info(
        "run=%03d started seed=%s initial_acc=%.4f audited_in=%s audited_out=%s target_client_size=%s",
        run_id,
        run_seed,
        initial_acc,
        int(np.sum(membership_flags)),
        int(len(membership_flags) - np.sum(membership_flags)),
        len(client_indices[0]),
    )
    append_jsonl(
        jsonl_path,
        {
            "event": "run_start",
            "run": run_id,
            "seed": run_seed,
            "audited_in": int(np.sum(membership_flags)),
            "audited_out": int(len(membership_flags) - np.sum(membership_flags)),
            "target_client_size": len(client_indices[0]),
        },
    )

    for round_id in range(1, args.rounds + 1):
        round_start = time.time()
        global_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        local_updates = []
        local_models = []
        local_losses = []
        start_acc = research.model_accuracy(global_model, combined_loader, device)
        LOGGER.info("run=%03d round=%03d start global_acc=%.4f", run_id, round_id, start_acc)

        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_state, local_loss = research.train_one_client(
                local_model,
                loader,
                device,
                args,
                run_id,
                round_id,
                client_id,
                jsonl_path,
            )
            parameter_names = [name for name, _ in local_model.named_parameters()]
            local_states.append(local_state)
            local_losses.append(local_loss)
            local_updates.append(research.state_update(global_state, local_state, parameter_names))
            local_models.append(local_model)

        cosine_rounds.append(
            research.measure_round(
                global_model,
                target_update=local_updates[0],
                reference_updates=local_updates[1:],
                samples=audited_samples,
                device=device,
                round_id=round_id,
                split="audited",
            )
        )
        loss_rounds.append(
            research.measure_loss_round(
                target_model=local_models[0],
                reference_models=local_models[1:],
                samples=audited_samples,
                device=device,
                round_id=round_id,
                split="audited",
            )
        )

        averaged_state = fed_avg_state_dicts(local_states, [len(dataset) for dataset in client_datasets])
        global_model.load_state_dict(averaged_state)
        del local_models
        end_acc = research.model_accuracy(global_model, combined_loader, device)
        round_elapsed = time.time() - round_start
        LOGGER.info(
            "run=%03d round=%03d end global_acc=%.4f mean_client_loss=%.6f "
            "audited_measurements=%s elapsed=%.2fs",
            run_id,
            round_id,
            end_acc,
            float(np.mean(local_losses)),
            len(audit_indices),
            round_elapsed,
        )
        append_jsonl(
            jsonl_path,
            {
                "event": "communication_round",
                "run": run_id,
                "round": round_id,
                "global_acc_start": start_acc,
                "global_acc_end": end_acc,
                "mean_client_loss": float(np.mean(local_losses)),
                "audited_measurements": len(audit_indices),
                "elapsed_seconds": round_elapsed,
            },
        )

    final_acc = research.model_accuracy(global_model, combined_loader, device)
    fedmia_config = FedMIAConfig(
        threshold=args.threshold,
        outlier_std_factor=args.outlier_std_factor,
        min_variance=args.min_variance,
    )
    scores_by_measurement = windowed_scores(cosine_rounds, loss_rounds, fedmia_config)
    metrics = summarize_scores(scores_by_measurement, membership_flags, args)
    elapsed = time.time() - run_start
    LOGGER.info(
        "run=%03d finished final_acc=%.4f "
        "cosine_auc=%.4f cosine_f1=%.4f loss_auc=%.4f loss_f1=%.4f "
        "combined_auc=%.4f combined_f1=%.4f elapsed=%.2fs",
        run_id,
        final_acc,
        metrics["cosine_auc"],
        metrics["cosine_f1"],
        metrics["loss_auc"],
        metrics["loss_f1"],
        metrics["combined_auc"],
        metrics["combined_f1"],
        elapsed,
    )

    for measurement_name, scores in scores_by_measurement.items():
        for audit_pos, patient_idx in enumerate(audit_indices):
            append_jsonl(
                jsonl_path,
                {
                    "event": "patient_score",
                    "run": run_id,
                    "measurement": measurement_name,
                    "patient_index": int(patient_idx),
                    "audit_position": int(audit_pos),
                    "membership": int(bool(membership_flags[audit_pos])),
                    "score": float(scores[audit_pos]),
                },
            )
    append_jsonl(
        jsonl_path,
        {
            "event": "run_result",
            "run": run_id,
            "seed": run_seed,
            "initial_acc": initial_acc,
            "final_acc": final_acc,
            "included_audited_patients": int(np.sum(membership_flags)),
            "elapsed_seconds": elapsed,
            **metrics,
        },
    )
    return RunSummary(
        run_id=run_id,
        seed=run_seed,
        initial_acc=initial_acc,
        final_acc=final_acc,
        included_audited_patients=int(np.sum(membership_flags)),
        elapsed_seconds=elapsed,
        metrics=metrics,
    ), scores_by_measurement


def add_scores_to_records(records, audit_indices, membership_flags, scores_by_measurement):
    for measurement_name, scores in scores_by_measurement.items():
        for audit_pos, patient_idx in enumerate(audit_indices):
            record = records[measurement_name][int(patient_idx)]
            record["scores"].append(float(scores[audit_pos]))
            record["labels"].append(int(bool(membership_flags[audit_pos])))


def patientwise_metrics(records, args):
    threshold_grid = research.parse_threshold_grid(args.threshold_grid)
    result = {}
    for measurement_name in MEASUREMENTS:
        rows = []
        pooled_member_scores = []
        pooled_nonmember_scores = []
        for patient_idx, record in records[measurement_name].items():
            member_scores = [
                score for score, label in zip(record["scores"], record["labels"]) if label == 1
            ]
            nonmember_scores = [
                score for score, label in zip(record["scores"], record["labels"]) if label == 0
            ]
            pooled_member_scores.extend(member_scores)
            pooled_nonmember_scores.extend(nonmember_scores)
            if member_scores and nonmember_scores:
                metrics, _ = research.score_list_metrics(
                    member_scores,
                    nonmember_scores,
                    args.threshold,
                    threshold_grid,
                )
                metrics["patient_index"] = patient_idx
                metrics["in_count"] = len(member_scores)
                metrics["out_count"] = len(nonmember_scores)
                rows.append(metrics)
        pooled_metrics, _ = research.score_list_metrics(
            pooled_member_scores,
            pooled_nonmember_scores,
            args.threshold,
            threshold_grid,
        )
        result[measurement_name] = {
            "patient_rows": rows,
            "pooled_metrics": pooled_metrics,
            "pooled_member_count": len(pooled_member_scores),
            "pooled_nonmember_count": len(pooled_nonmember_scores),
        }
    return result


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1 if len(arr) > 1 else 0))


def build_report(
    experiment_id,
    args,
    graph,
    selected_features,
    audit_indices,
    membership_matrix,
    run_summaries,
    metrics_by_measurement,
    log_path,
    jsonl_path,
    total_elapsed,
):
    metric_names = [
        "tpr",
        "tnr",
        "fpr",
        "fnr",
        "precision",
        "recall",
        "f1",
        "threshold_accuracy",
        "auc",
        "log_auc",
        "tpr_at_fpr_0.1",
        "tpr_at_fpr_0.01",
        "best_f1",
        "best_f1_threshold",
        "best_f1_tpr",
        "best_f1_tnr",
    ]
    lines = [
        "FedMIA BINN LiRA-Style Patient Protocol Report",
        f"experiment_id: {experiment_id}",
        "script: experiments/fedmia_binn_lira_protocol.py",
        f"log_file: {log_path}",
        f"jsonl_log_file: {jsonl_path}",
        "",
        "Protocol",
        "  Unit of repetition: one full federated BINN training trajectory.",
        "  Audited-patient protocol: each audited patient is independently IN with probability p.",
        "  IN means the patient is placed in the target client; OUT means absent from all clients.",
        "  Non-target clients are sampled from non-audited background patients only.",
        "  Aggregation: per-patient attack metrics across trajectories, then mean +/- sample std.",
        "",
        "Data And DAG",
        f"  x_file: data/datasets/ProstateCancer/{args.x_file}",
        f"  y_file: data/datasets/ProstateCancer/{args.y_file}",
        f"  samples_loaded: {args.max_samples if args.max_samples else 'all'}",
        f"  selected_reactome_mapped_features: {len(selected_features)}",
        f"  graph_nodes: {graph.number_of_nodes()}",
        f"  graph_edges: {graph.number_of_edges()}",
        "",
        "Configuration",
        f"  runs: {args.runs}",
        f"  audit_count: {args.audit_count}",
        f"  inclusion_probability: {args.inclusion_prob}",
        f"  clients: {args.num_clients}",
        f"  rounds: {args.rounds}",
        f"  local_epochs: {args.local_epochs}",
        f"  samples_per_client: {args.samples_per_client}",
        f"  score_windows: {', '.join(WINDOWS)}",
        f"  threshold_delta: {args.threshold}",
        f"  threshold_grid: {args.threshold_grid}",
        f"  device_requested: {args.device}",
        f"  gpu: {args.gpu}",
        f"  total_elapsed_seconds: {total_elapsed:.2f}",
        "",
        "Membership Matrix",
        f"  total_in_assignments: {int(membership_matrix.sum())}",
        f"  total_out_assignments: {int(membership_matrix.size - membership_matrix.sum())}",
        f"  per_patient_in_count_mean: {float(membership_matrix.sum(axis=0).mean()):.4f}",
        f"  audited_patient_indices_head: {list(map(int, audit_indices[:10]))}",
        "",
        "Aggregated Patient-Wise Performance Metrics",
    ]
    for measurement_name in PRIMARY_MEASUREMENTS:
        rows = metrics_by_measurement[measurement_name]["patient_rows"]
        pooled = metrics_by_measurement[measurement_name]["pooled_metrics"]
        lines.append(f"  [{measurement_name}]")
        lines.append(f"    patients_with_in_and_out: {len(rows)}")
        for metric_name in metric_names:
            mean, std = mean_std([row[metric_name] for row in rows])
            lines.append(f"    patient_mean_{metric_name}: {mean:.6f} +/- {std:.6f}")
        lines.append(
            "    pooled: "
            f"auc={pooled['auc']:.6f} f1={pooled['f1']:.6f} "
            f"tpr={pooled['tpr']:.6f} tnr={pooled['tnr']:.6f} "
            f"best_f1={pooled['best_f1']:.6f} best_threshold={pooled['best_f1_threshold']:.6f}"
        )

    window_metric_names = [
        "auc",
        "f1",
        "tpr",
        "tnr",
        "tpr_at_fpr_0.1",
        "tpr_at_fpr_0.01",
        "best_f1",
        "best_f1_threshold",
    ]
    lines.extend(["", "Round Window Diagnostics"])
    for measurement_name in MEASUREMENTS:
        if measurement_name in PRIMARY_MEASUREMENTS:
            continue
        rows = metrics_by_measurement[measurement_name]["patient_rows"]
        pooled = metrics_by_measurement[measurement_name]["pooled_metrics"]
        lines.append(f"  [{measurement_name}]")
        for metric_name in window_metric_names:
            mean, std = mean_std([row[metric_name] for row in rows])
            lines.append(f"    patient_mean_{metric_name}: {mean:.6f} +/- {std:.6f}")
        lines.append(
            "    pooled: "
            f"auc={pooled['auc']:.6f} f1={pooled['f1']:.6f} "
            f"tpr={pooled['tpr']:.6f} tnr={pooled['tnr']:.6f} "
            f"best_f1={pooled['best_f1']:.6f} best_threshold={pooled['best_f1_threshold']:.6f}"
        )

    init_mean, init_std = mean_std([summary.initial_acc for summary in run_summaries])
    final_mean, final_std = mean_std([summary.final_acc for summary in run_summaries])
    lines.extend(
        [
            "",
            "Training Accuracy Across Trajectories",
            f"  initial_acc: {init_mean:.6f} +/- {init_std:.6f}",
            f"  final_acc: {final_mean:.6f} +/- {final_std:.6f}",
            "",
            "Per-Run Summary",
        ]
    )
    for summary in run_summaries:
        lines.append(
            f"  run={summary.run_id:03d} seed={summary.seed} audited_in={summary.included_audited_patients} "
            f"cos_auc={summary.metrics['cosine_auc']:.6f} loss_auc={summary.metrics['loss_auc']:.6f} "
            f"comb_auc={summary.metrics['combined_auc']:.6f} final_acc={summary.final_acc:.6f} "
            f"elapsed={summary.elapsed_seconds:.2f}s"
        )

    lines.extend(
        [
            "",
            "Commentary",
            "  This protocol is designed to mirror the central BINN report's repeated patient-wise",
            "  IN/OUT evaluation, while replacing shadow central models with repeated federated",
            "  trajectories and FedMIA's non-target-client Qout estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    experiment_id = make_experiment_id()
    log_path, jsonl_path = configure_logging(args, experiment_id)
    total_start = time.time()
    set_seed(args.seed)
    device = resolve_device(args)
    LOGGER.info("experiment_id=%s", experiment_id)
    LOGGER.info("device=%s cuda_available=%s", device, torch.cuda.is_available())
    LOGGER.info("arguments=%s", vars(args))

    feature_index = research.load_pnet_feature_index()
    graph, selected_features = research.build_real_reactome_feature_graph(feature_index, args.feature_limit)
    x, y = research.load_pnet_data(selected_features, args)
    rng = np.random.default_rng(args.seed)
    audit_indices = balanced_sample_from_pool(y, args.audit_count, rng, list(range(y.shape[0])))
    membership_matrix = make_membership_matrix(
        args.runs,
        args.audit_count,
        args.inclusion_prob,
        rng,
    )
    LOGGER.info(
        "loaded data samples=%s features=%s positive_rate=%.4f audited_patients=%s",
        x.shape[0],
        x.shape[1],
        float(y.mean().item()),
        len(audit_indices),
    )
    LOGGER.info("real_dag nodes=%s edges=%s", graph.number_of_nodes(), graph.number_of_edges())

    records = {name: defaultdict(lambda: {"scores": [], "labels": []}) for name in MEASUREMENTS}
    run_summaries = []
    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx * 1009
        membership_flags = membership_matrix[run_idx - 1]
        summary, scores_by_measurement = run_one_trajectory(
            run_idx,
            run_seed,
            graph,
            x,
            y,
            audit_indices,
            membership_flags,
            args,
            device,
            jsonl_path,
        )
        add_scores_to_records(records, audit_indices, membership_flags, scores_by_measurement)
        run_summaries.append(summary)

    metrics_by_measurement = patientwise_metrics(records, args)
    total_elapsed = time.time() - total_start
    report_text = build_report(
        experiment_id,
        args,
        graph,
        selected_features,
        audit_indices,
        membership_matrix,
        run_summaries,
        metrics_by_measurement,
        log_path,
        jsonl_path,
        total_elapsed,
    )
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, f"{experiment_id}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    print(report_text)
    print(f"report_path: {report_path}")
    LOGGER.info("report_path=%s", report_path)


if __name__ == "__main__":
    main()
