from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import FedMIAConfig, FedMIARoundMeasurements, evaluate_membership  # noqa: E402
from models.binn import binn  # noqa: E402
from utils.federated import fed_avg_state_dicts  # noqa: E402

import experiments.fedmia_binn_research as research  # noqa: E402


LOGGER = logging.getLogger("fedmia_binn_paper_grid")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
LOG_DIR = os.path.join(REPO_ROOT, "logs")


@dataclass(frozen=True)
class GridConfig:
    config_id: int
    clients: int
    rounds: int
    local_epochs: int
    beta: Optional[float]
    beta_label: str
    sample_fraction: Optional[float]
    samples_per_client: int


@dataclass
class TrajectoryResult:
    config: GridConfig
    run_id: int
    seed: int
    train_acc_initial: float
    train_acc_final: float
    holdout_acc_initial: float
    holdout_acc_final: float
    member_count: int
    nonmember_count: int
    client_sizes: List[int]
    elapsed_seconds: float
    metrics: Dict[str, float]
    member_scores: Dict[str, List[float]]
    nonmember_scores: Dict[str, List[float]]
    member_indices: List[int]
    nonmember_indices: List[int]
    nonmember_source: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedMIA paper-style grid evaluation on federated BINN."
    )
    parser.add_argument("--runs", type=int, default=3, help="independent FL trajectories per grid cell")
    parser.add_argument("--client-grid", default="10", help="comma-separated client counts")
    parser.add_argument("--round-grid", default="300", help="comma-separated communication rounds")
    parser.add_argument("--local-epoch-grid", default="1", help="comma-separated local epoch counts")
    parser.add_argument(
        "--beta-grid",
        default="iid",
        help="comma-separated Non-IID Dirichlet beta values; use iid for beta=infinity",
    )
    parser.add_argument(
        "--sample-fraction-grid",
        default="1.0",
        help="comma-separated fractions of BINN's maximum disjoint samples/client",
    )
    parser.add_argument(
        "--samples-per-client-grid",
        default="auto",
        help="comma-separated absolute samples/client, or auto to use sample fractions",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.1,
        help="one-tenth holdout by default, matching the FedMIA appendix nonmember pool",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help="candidate members/nonmembers per run; 0 uses the largest balanced target/nonmember set",
    )
    parser.add_argument(
        "--nonmember-source",
        default="holdout",
        choices=["holdout", "other_clients", "target_nonmembers", "combined"],
        help=(
            "OUT candidate source relative to target client 0: holdout uses only globally unseen patients; "
            "other_clients uses only patients trained by non-target clients; "
            "target_nonmembers uses both non-target-client patients and holdout patients. "
            "combined is a legacy alias for target_nonmembers."
        ),
    )
    parser.add_argument(
        "--min-patient-state-appearances",
        type=int,
        default=2,
        help="minimum IN and OUT appearances required for patientwise vulnerability ranking",
    )
    parser.add_argument(
        "--audit-patient-count",
        type=int,
        default=0,
        help=(
            "fixed audited-patient count; 0 keeps random candidate sampling. "
            "When >0, audited patients are alternated between target-client IN and configured OUT states."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16, help="local training batch size")
    parser.add_argument("--lr", type=float, default=0.03, help="local optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="local optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--threshold", type=float, default=0.5, help="FedMIA delta threshold")
    parser.add_argument(
        "--threshold-grid",
        default="0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="comma-separated deltas used for best-F1 diagnostics",
    )
    parser.add_argument("--outlier-std-factor", type=float, default=3.0, help="FedMIA sigma filter")
    parser.add_argument("--min-variance", type=float, default=1e-8, help="FedMIA variance floor")
    parser.add_argument("--seed", type=int, default=20260509, help="base random seed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="training device")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--x-file", default="pnet_x.npy", help="ProstateCancer feature array")
    parser.add_argument("--y-file", default="pnet_y.npy", help="ProstateCancer label array")
    parser.add_argument("--max-samples", type=int, default=0, help="debug sample cap; 0 uses all")
    parser.add_argument("--feature-limit", type=int, default=0, help="debug feature cap; 0 uses all")
    parser.add_argument("--max-configs", type=int, default=0, help="debug cap on grid cells; 0 uses all")
    parser.add_argument("--report-dir", default=REPORT_DIR, help="directory for reports")
    parser.add_argument("--log-dir", default=LOG_DIR, help="directory for logs")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def make_experiment_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"fedmia_binn_paper_grid_{stamp}_{uuid.uuid4().hex[:8]}"


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


def parse_int_grid(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_beta_grid(value: str) -> List[Tuple[str, Optional[float]]]:
    betas = []
    for part in value.split(","):
        label = part.strip()
        if not label:
            continue
        if label.lower() in {"iid", "inf", "infinity"}:
            betas.append(("iid", None))
        else:
            betas.append((label, float(label)))
    return betas


def tensor_state_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


def labels_numpy(labels: torch.Tensor) -> np.ndarray:
    return labels.view(-1).detach().cpu().numpy().astype(int)


def stratified_holdout_split(labels: torch.Tensor, holdout_fraction: float, rng) -> Tuple[List[int], List[int]]:
    label_values = labels_numpy(labels)
    train_indices: List[int] = []
    holdout_indices: List[int] = []
    for label in sorted(np.unique(label_values)):
        indices = np.where(label_values == label)[0].tolist()
        rng.shuffle(indices)
        holdout_count = int(round(len(indices) * holdout_fraction))
        if holdout_fraction > 0.0 and len(indices) > 1:
            holdout_count = min(max(1, holdout_count), len(indices) - 1)
        holdout_indices.extend(indices[:holdout_count])
        train_indices.extend(indices[holdout_count:])
    rng.shuffle(train_indices)
    rng.shuffle(holdout_indices)
    return train_indices, holdout_indices


def balanced_draw(labels: torch.Tensor, count: int, rng, pool: Iterable[int], exclude=None) -> List[int]:
    if count <= 0:
        return []
    exclude = set(exclude or [])
    label_values = labels_numpy(labels)
    available = [idx for idx in pool if idx not in exclude]
    classes = sorted(np.unique(label_values[available]).tolist())
    selected: List[int] = []
    per_class = count // max(len(classes), 1)
    for label in classes:
        candidates = [idx for idx in available if label_values[idx] == label and idx not in selected]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    if len(selected) < count:
        remaining = [idx for idx in available if idx not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"Not enough samples to draw {count} records from pool of {len(available)}.")
    rng.shuffle(selected)
    return selected[:count]


def make_iid_clients(labels, train_indices, clients: int, samples_per_client: int, rng) -> List[List[int]]:
    used = set()
    client_indices = []
    for _ in range(clients):
        indices = balanced_draw(labels, samples_per_client, rng, train_indices, used)
        used.update(indices)
        client_indices.append(indices)
    return client_indices


def fix_empty_clients(client_indices: List[List[int]], rng) -> List[List[int]]:
    for client_id, indices in enumerate(client_indices):
        if indices:
            continue
        donor_id = max(range(len(client_indices)), key=lambda idx: len(client_indices[idx]))
        if len(client_indices[donor_id]) <= 1:
            raise ValueError("Dirichlet split produced empty clients and no donor has spare samples.")
        move_at = int(rng.integers(0, len(client_indices[donor_id])))
        indices.append(client_indices[donor_id].pop(move_at))
        LOGGER.warning("Dirichlet split produced empty client=%s; moved one sample from client=%s.", client_id, donor_id)
    return client_indices


def make_dirichlet_clients(labels, train_indices, clients: int, samples_per_client: int, beta: float, rng) -> List[List[int]]:
    total_count = clients * samples_per_client
    selected_pool = balanced_draw(labels, total_count, rng, train_indices)
    label_values = labels_numpy(labels)
    client_indices: List[List[int]] = [[] for _ in range(clients)]
    for label in sorted(np.unique(label_values[selected_pool]).tolist()):
        class_indices = [idx for idx in selected_pool if label_values[idx] == label]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(beta, clients))
        raw_counts = proportions * len(class_indices)
        counts = np.floor(raw_counts).astype(int)
        remainder = len(class_indices) - int(counts.sum())
        if remainder:
            order = np.argsort(raw_counts - counts)[::-1]
            for client_id in order[:remainder]:
                counts[client_id] += 1
        start = 0
        for client_id, count in enumerate(counts.tolist()):
            client_indices[client_id].extend(class_indices[start : start + count])
            start += count
    for indices in client_indices:
        rng.shuffle(indices)
    return fix_empty_clients(client_indices, rng)


def make_client_indices(labels, train_indices, config: GridConfig, rng) -> List[List[int]]:
    if config.beta is None:
        return make_iid_clients(labels, train_indices, config.clients, config.samples_per_client, rng)
    return make_dirichlet_clients(
        labels,
        train_indices,
        config.clients,
        config.samples_per_client,
        config.beta,
        rng,
    )


def nonmember_pool_for_source(client_indices, holdout_indices, nonmember_source: str) -> List[int]:
    if nonmember_source == "holdout":
        return list(holdout_indices)
    other_client_indices = [
        idx for client_indices_for_one_client in client_indices[1:]
        for idx in client_indices_for_one_client
    ]
    if nonmember_source == "other_clients":
        return other_client_indices
    if nonmember_source in {"target_nonmembers", "combined"}:
        return other_client_indices + list(holdout_indices)
    raise ValueError(f"Unsupported nonmember source: {nonmember_source}")


def make_candidate_indices(
    labels,
    client_indices,
    holdout_indices,
    candidate_count: int,
    rng,
    nonmember_source: str = "holdout",
):
    target_pool = list(client_indices[0])
    nonmember_pool = nonmember_pool_for_source(client_indices, holdout_indices, nonmember_source)
    if not nonmember_pool:
        raise ValueError(f"Nonmember source '{nonmember_source}' produced an empty candidate pool.")
    max_balanced_count = min(len(target_pool), len(nonmember_pool))
    member_count = max_balanced_count if candidate_count <= 0 else min(candidate_count, max_balanced_count)
    if member_count < len(target_pool) and candidate_count <= 0:
        LOGGER.info(
            "Capping attack candidates to %s because target client has %s samples and %s has %s.",
            member_count,
            len(target_pool),
            nonmember_source,
            len(nonmember_pool),
        )
    member_indices = balanced_draw(labels, member_count, rng, target_pool)
    nonmember_indices = balanced_draw(labels, member_count, rng, nonmember_pool)
    return member_indices, nonmember_indices


def choose_audit_patient_indices(labels, train_indices, audit_patient_count: int, rng) -> List[int]:
    if audit_patient_count <= 0:
        return []
    count = min(int(audit_patient_count), len(train_indices))
    return balanced_draw(labels, count, rng, train_indices)


def apply_audit_patient_assignments(
    client_indices: Sequence[Sequence[int]],
    audit_indices: Sequence[int],
    run_id: int,
    rng,
    nonmember_source: str = "target_nonmembers",
) -> Tuple[List[List[int]], List[int], List[int]]:
    if not audit_indices:
        return [list(indices) for indices in client_indices], [], []
    if nonmember_source not in {"holdout", "other_clients", "target_nonmembers", "combined"}:
        raise ValueError(f"Unsupported audited-patient nonmember source: {nonmember_source}")
    if nonmember_source == "other_clients" and len(client_indices) < 2:
        raise ValueError("Audited-patient OUT assignment requires at least two clients.")

    audit_set = {int(idx) for idx in audit_indices}
    cleaned_clients = [
        [int(idx) for idx in indices if int(idx) not in audit_set]
        for indices in client_indices
    ]
    member_indices: List[int] = []
    nonmember_indices: List[int] = []

    for position, patient_index in enumerate(int(idx) for idx in audit_indices):
        is_member = ((position + run_id) % 2) == 0
        if is_member:
            cleaned_clients[0].append(patient_index)
            member_indices.append(patient_index)
        else:
            use_other_client = nonmember_source == "other_clients" or (
                nonmember_source in {"target_nonmembers", "combined"}
                and len(cleaned_clients) > 1
                and float(rng.random()) < 0.5
            )
            if use_other_client:
                non_target_client = 1 + int(rng.integers(0, len(cleaned_clients) - 1))
                cleaned_clients[non_target_client].append(patient_index)
            nonmember_indices.append(patient_index)

    for indices in cleaned_clients:
        rng.shuffle(indices)
    rng.shuffle(member_indices)
    rng.shuffle(nonmember_indices)
    return cleaned_clients, member_indices, nonmember_indices


def make_loaders(x, y, client_indices, holdout_indices, args, seed):
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
    train_dataset = TensorDataset(
        torch.cat([dataset.tensors[0] for dataset in client_datasets], dim=0),
        torch.cat([dataset.tensors[1] for dataset in client_datasets], dim=0),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    holdout_loader = DataLoader(
        TensorDataset(x[holdout_indices], y[holdout_indices]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return client_datasets, client_loaders, train_loader, holdout_loader


def fpr_tpr_from_scores(member_scores: Sequence[float], nonmember_scores: Sequence[float]):
    return research.roc_curve_from_scores(member_scores, nonmember_scores)


def tpr_at_fpr(member_scores: Sequence[float], nonmember_scores: Sequence[float], threshold: float) -> float:
    fprs, tprs = fpr_tpr_from_scores(member_scores, nonmember_scores)
    best = 0.0
    for fpr, tpr in zip(fprs, tprs):
        if fpr < threshold:
            best = tpr
    return best


def metrics_for_evaluation(evaluation, threshold: float, threshold_grid: Sequence[float]):
    metrics, sweep_rows = research.evaluation_metrics(evaluation, threshold, threshold_grid)
    metrics["auc"] = evaluation.auc
    metrics["log_auc"] = evaluation.log_auc
    metrics["tpr_at_fpr_0.001"] = evaluation.tprs.get("0.001", 0.0)
    metrics["tpr_at_fpr_0.0001"] = evaluation.tprs.get("0.0001", 0.0)
    return metrics, sweep_rows


def run_one_trajectory(
    config: GridConfig,
    run_id: int,
    run_seed: int,
    graph,
    x,
    y,
    train_indices,
    holdout_indices,
    audit_indices,
    args,
    device,
    jsonl_path: str,
) -> TrajectoryResult:
    start_time = time.time()
    set_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    client_indices = make_client_indices(y, train_indices, config, rng)
    if audit_indices:
        client_indices, member_indices, nonmember_indices = apply_audit_patient_assignments(
            client_indices,
            audit_indices,
            run_id,
            rng,
            args.nonmember_source,
        )
    else:
        member_indices, nonmember_indices = make_candidate_indices(
            y,
            client_indices,
            holdout_indices,
            args.candidate_count,
            rng,
            args.nonmember_source,
        )
    client_datasets, client_loaders, train_loader, holdout_loader = make_loaders(
        x, y, client_indices, holdout_indices, args, run_seed
    )
    member_samples = [(x[idx], y[idx]) for idx in member_indices]
    nonmember_samples = [(x[idx], y[idx]) for idx in nonmember_indices]
    run_args = argparse.Namespace(**vars(args))
    run_args.local_epochs = config.local_epochs

    global_model = binn(graph=graph, output_size=1, dropout_prob=0.0, output_last_layers=1).to(device)
    train_acc_initial = research.model_accuracy(global_model, train_loader, device)
    holdout_acc_initial = research.model_accuracy(global_model, holdout_loader, device)
    cosine_member_rounds: List[FedMIARoundMeasurements] = []
    cosine_nonmember_rounds: List[FedMIARoundMeasurements] = []
    loss_member_rounds: List[FedMIARoundMeasurements] = []
    loss_nonmember_rounds: List[FedMIARoundMeasurements] = []

    LOGGER.info(
        "config=%03d run=%03d started seed=%s clients=%s rounds=%s local_epochs=%s "
        "beta=%s samples_per_client=%s member_candidates=%s nonmember_candidates=%s "
        "nonmember_source=%s initial_train_acc=%.4f initial_holdout_acc=%.4f client_sizes=%s",
        config.config_id,
        run_id,
        run_seed,
        config.clients,
        config.rounds,
        config.local_epochs,
        config.beta_label,
        config.samples_per_client,
        len(member_indices),
        len(nonmember_indices),
        args.nonmember_source,
        train_acc_initial,
        holdout_acc_initial,
        [len(indices) for indices in client_indices],
    )
    append_jsonl(
        jsonl_path,
        {
            "event": "run_start",
            "config_id": config.config_id,
            "run": run_id,
            "seed": run_seed,
            "nonmember_source": args.nonmember_source,
            "audit_patient_count": len(audit_indices),
            "client_sizes": [len(indices) for indices in client_indices],
            "member_candidates": len(member_indices),
            "nonmember_candidates": len(nonmember_indices),
            "train_acc_initial": train_acc_initial,
            "holdout_acc_initial": holdout_acc_initial,
        },
    )

    for round_id in range(1, config.rounds + 1):
        round_start = time.time()
        global_state = tensor_state_to_cpu(global_model.state_dict())
        local_states = []
        local_updates = []
        local_losses = []
        loss_member_target = None
        loss_nonmember_target = None
        loss_member_refs = []
        loss_nonmember_refs = []
        round_train_acc_start = research.model_accuracy(global_model, train_loader, device)
        LOGGER.info(
            "config=%03d run=%03d round=%03d start train_acc=%.4f",
            config.config_id,
            run_id,
            round_id,
            round_train_acc_start,
        )

        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_state, local_loss = research.train_one_client(
                local_model,
                loader,
                device,
                run_args,
                run_id,
                round_id,
                client_id,
                jsonl_path,
            )
            local_state = tensor_state_to_cpu(local_state)
            parameter_names = [name for name, _ in local_model.named_parameters()]
            local_update = research.state_update(global_state, local_state, parameter_names)
            member_loss_values = research.negative_loss_measurements(local_model, member_samples, device)
            nonmember_loss_values = research.negative_loss_measurements(local_model, nonmember_samples, device)
            if client_id == 0:
                loss_member_target = member_loss_values
                loss_nonmember_target = nonmember_loss_values
            else:
                loss_member_refs.append(member_loss_values)
                loss_nonmember_refs.append(nonmember_loss_values)
            local_states.append(local_state)
            local_updates.append(local_update)
            local_losses.append(local_loss)
            del local_model

        cosine_member_rounds.append(
            research.measure_round(
                global_model,
                target_update=local_updates[0],
                reference_updates=local_updates[1:],
                samples=member_samples,
                device=device,
                round_id=round_id,
                split="member",
            )
        )
        cosine_nonmember_rounds.append(
            research.measure_round(
                global_model,
                target_update=local_updates[0],
                reference_updates=local_updates[1:],
                samples=nonmember_samples,
                device=device,
                round_id=round_id,
                split="nonmember",
            )
        )
        loss_member_rounds.append(
            FedMIARoundMeasurements(
                target_measurements=loss_member_target or [],
                reference_measurements=loss_member_refs,
                round_id=round_id,
                metadata={"split": "member", "measurement": "negative_loss"},
            )
        )
        loss_nonmember_rounds.append(
            FedMIARoundMeasurements(
                target_measurements=loss_nonmember_target or [],
                reference_measurements=loss_nonmember_refs,
                round_id=round_id,
                metadata={"split": "nonmember", "measurement": "negative_loss"},
            )
        )

        averaged_state = fed_avg_state_dicts(local_states, [len(dataset) for dataset in client_datasets])
        global_model.load_state_dict(averaged_state)
        round_train_acc_end = research.model_accuracy(global_model, train_loader, device)
        round_holdout_acc_end = research.model_accuracy(global_model, holdout_loader, device)
        round_elapsed = time.time() - round_start
        LOGGER.info(
            "config=%03d run=%03d round=%03d end train_acc=%.4f holdout_acc=%.4f "
            "mean_client_loss=%.6f elapsed=%.2fs",
            config.config_id,
            run_id,
            round_id,
            round_train_acc_end,
            round_holdout_acc_end,
            float(np.mean(local_losses)),
            round_elapsed,
        )
        append_jsonl(
            jsonl_path,
            {
                "event": "communication_round",
                "config_id": config.config_id,
                "run": run_id,
                "round": round_id,
                "train_acc_start": round_train_acc_start,
                "train_acc_end": round_train_acc_end,
                "holdout_acc_end": round_holdout_acc_end,
                "mean_client_loss": float(np.mean(local_losses)),
                "elapsed_seconds": round_elapsed,
            },
        )

    train_acc_final = research.model_accuracy(global_model, train_loader, device)
    holdout_acc_final = research.model_accuracy(global_model, holdout_loader, device)
    fedmia_config = FedMIAConfig(
        threshold=args.threshold,
        outlier_std_factor=args.outlier_std_factor,
        min_variance=args.min_variance,
    )
    threshold_grid = research.parse_threshold_grid(args.threshold_grid)
    loss_evaluation = evaluate_membership(loss_member_rounds, loss_nonmember_rounds, fedmia_config)
    cosine_evaluation = evaluate_membership(cosine_member_rounds, cosine_nonmember_rounds, fedmia_config)
    loss_metrics, loss_sweep_rows = metrics_for_evaluation(
        loss_evaluation, args.threshold, threshold_grid
    )
    cosine_metrics, cosine_sweep_rows = metrics_for_evaluation(
        cosine_evaluation, args.threshold, threshold_grid
    )
    metrics = {}
    metrics.update(research.prefix_metrics("fedmia_i_loss", loss_metrics))
    metrics.update(research.prefix_metrics("fedmia_ii_cosine", cosine_metrics))
    elapsed = time.time() - start_time
    LOGGER.info(
        "config=%03d run=%03d finished train_acc=%.4f holdout_acc=%.4f "
        "fedmia_i_auc=%.4f fedmia_i_tpr_at_fpr_0.001=%.4f fedmia_i_f1=%.4f "
        "fedmia_ii_auc=%.4f fedmia_ii_tpr_at_fpr_0.001=%.4f fedmia_ii_f1=%.4f "
        "elapsed=%.2fs",
        config.config_id,
        run_id,
        train_acc_final,
        holdout_acc_final,
        metrics["fedmia_i_loss_auc"],
        metrics["fedmia_i_loss_tpr_at_fpr_0.001"],
        metrics["fedmia_i_loss_f1"],
        metrics["fedmia_ii_cosine_auc"],
        metrics["fedmia_ii_cosine_tpr_at_fpr_0.001"],
        metrics["fedmia_ii_cosine_f1"],
        elapsed,
    )
    append_jsonl(
        jsonl_path,
        {
            "event": "run_result",
            "config_id": config.config_id,
            "run": run_id,
            "seed": run_seed,
            "train_acc_initial": train_acc_initial,
            "train_acc_final": train_acc_final,
            "holdout_acc_initial": holdout_acc_initial,
            "holdout_acc_final": holdout_acc_final,
            "member_count": len(member_indices),
            "nonmember_count": len(nonmember_indices),
            "nonmember_source": args.nonmember_source,
            "audit_patient_count": len(audit_indices),
            "client_sizes": [len(indices) for indices in client_indices],
            "elapsed_seconds": elapsed,
            "fedmia_i_loss_threshold_sweep": loss_sweep_rows,
            "fedmia_ii_cosine_threshold_sweep": cosine_sweep_rows,
            **metrics,
        },
    )
    return TrajectoryResult(
        config=config,
        run_id=run_id,
        seed=run_seed,
        train_acc_initial=train_acc_initial,
        train_acc_final=train_acc_final,
        holdout_acc_initial=holdout_acc_initial,
        holdout_acc_final=holdout_acc_final,
        member_count=len(member_indices),
        nonmember_count=len(nonmember_indices),
        client_sizes=[len(indices) for indices in client_indices],
        elapsed_seconds=elapsed,
        metrics=metrics,
        member_scores={
            "fedmia_i_loss": loss_evaluation.member_scores.aggregate_scores,
            "fedmia_ii_cosine": cosine_evaluation.member_scores.aggregate_scores,
        },
        nonmember_scores={
            "fedmia_i_loss": loss_evaluation.nonmember_scores.aggregate_scores,
            "fedmia_ii_cosine": cosine_evaluation.nonmember_scores.aggregate_scores,
        },
        member_indices=[int(idx) for idx in member_indices],
        nonmember_indices=[int(idx) for idx in nonmember_indices],
        nonmember_source=args.nonmember_source,
    )


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1 if len(arr) > 1 else 0))


def pooled_metrics(results: Sequence[TrajectoryResult], measurement: str, args) -> Dict[str, float]:
    member_scores: List[float] = []
    nonmember_scores: List[float] = []
    for result in results:
        member_scores.extend(result.member_scores[measurement])
        nonmember_scores.extend(result.nonmember_scores[measurement])
    metrics, _ = research.score_list_metrics(
        member_scores,
        nonmember_scores,
        args.threshold,
        research.parse_threshold_grid(args.threshold_grid),
    )
    metrics["tpr_at_fpr_0.001"] = tpr_at_fpr(member_scores, nonmember_scores, 0.001)
    metrics["tpr_at_fpr_0.0001"] = tpr_at_fpr(member_scores, nonmember_scores, 0.0001)
    metrics["member_count"] = float(len(member_scores))
    metrics["nonmember_count"] = float(len(nonmember_scores))
    return metrics


def auc_contribution_for_members(member_scores: Sequence[float], nonmember_scores: Sequence[float]) -> List[float]:
    contributions = []
    for member_score in member_scores:
        wins = sum(1.0 for score in nonmember_scores if member_score > score)
        ties = sum(1.0 for score in nonmember_scores if member_score == score)
        contributions.append((wins + 0.5 * ties) / max(len(nonmember_scores), 1))
    return contributions


def auc_contribution_for_nonmembers(member_scores: Sequence[float], nonmember_scores: Sequence[float]) -> List[float]:
    contributions = []
    for nonmember_score in nonmember_scores:
        wins = sum(1.0 for score in member_scores if score > nonmember_score)
        ties = sum(1.0 for score in member_scores if score == nonmember_score)
        contributions.append((wins + 0.5 * ties) / max(len(member_scores), 1))
    return contributions


def safe_mean(values: Sequence[float]) -> float:
    clean_values = [float(value) for value in values if not math.isnan(float(value))]
    if not clean_values:
        return float("nan")
    return float(np.mean(clean_values))


def safe_sample_std(values: Sequence[float]) -> float:
    clean_values = [float(value) for value in values if not math.isnan(float(value))]
    if len(clean_values) <= 1:
        return 0.0
    return float(np.std(clean_values, ddof=1))


def make_patient_observation_rows(results: Sequence[TrajectoryResult], args, y) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for result in results:
        for measurement in ("fedmia_i_loss", "fedmia_ii_cosine"):
            member_scores = result.member_scores[measurement]
            nonmember_scores = result.nonmember_scores[measurement]
            member_auc_parts = auc_contribution_for_members(member_scores, nonmember_scores)
            nonmember_auc_parts = auc_contribution_for_nonmembers(member_scores, nonmember_scores)
            for patient_index, score, auc_part in zip(
                result.member_indices, member_scores, member_auc_parts
            ):
                prediction = 1 if score > args.threshold else 0
                rows.append(
                    {
                        "config_id": result.config.config_id,
                        "run": result.run_id,
                        "seed": result.seed,
                        "measurement": measurement,
                        "nonmember_source": result.nonmember_source,
                        "patient_index": int(patient_index),
                        "class_label": int(y[patient_index].item()),
                        "membership_label": 1,
                        "prediction": prediction,
                        "correct": 1 if prediction == 1 else 0,
                        "score": float(score),
                        "auc_contribution": float(auc_part),
                    }
                )
            for patient_index, score, auc_part in zip(
                result.nonmember_indices, nonmember_scores, nonmember_auc_parts
            ):
                prediction = 1 if score > args.threshold else 0
                rows.append(
                    {
                        "config_id": result.config.config_id,
                        "run": result.run_id,
                        "seed": result.seed,
                        "measurement": measurement,
                        "nonmember_source": result.nonmember_source,
                        "patient_index": int(patient_index),
                        "class_label": int(y[patient_index].item()),
                        "membership_label": 0,
                        "prediction": prediction,
                        "correct": 1 if prediction == 0 else 0,
                        "score": float(score),
                        "auc_contribution": float(auc_part),
                    }
                )
    return rows


def patient_f1(tp: int, fp: int, fn: int) -> float:
    if tp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def patient_auc(member_scores: Sequence[float], nonmember_scores: Sequence[float]) -> float:
    if not member_scores or not nonmember_scores:
        return float("nan")
    fprs, tprs = research.roc_curve_from_scores(member_scores, nonmember_scores)
    return research.trapezoid_auc(fprs, tprs)


def make_patient_metric_rows(observation_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[int, str, int], List[Dict[str, float]]] = {}
    for row in observation_rows:
        key = (int(row["config_id"]), str(row["measurement"]), int(row["patient_index"]))
        grouped.setdefault(key, []).append(row)

    metric_rows: List[Dict[str, float]] = []
    for (config_id, measurement, patient_index), rows in sorted(grouped.items()):
        member_rows = [row for row in rows if int(row["membership_label"]) == 1]
        nonmember_rows = [row for row in rows if int(row["membership_label"]) == 0]
        tp = sum(1 for row in member_rows if int(row["prediction"]) == 1)
        fn = len(member_rows) - tp
        tn = sum(1 for row in nonmember_rows if int(row["prediction"]) == 0)
        fp = len(nonmember_rows) - tn
        member_scores = [float(row["score"]) for row in member_rows]
        nonmember_scores = [float(row["score"]) for row in nonmember_rows]
        member_auc_parts = [float(row["auc_contribution"]) for row in member_rows]
        nonmember_auc_parts = [float(row["auc_contribution"]) for row in nonmember_rows]
        all_scores = [float(row["score"]) for row in rows]
        all_auc_parts = [float(row["auc_contribution"]) for row in rows]
        member_score_mean = safe_mean(member_scores)
        nonmember_score_mean = safe_mean(nonmember_scores)
        score_gap = (
            member_score_mean - nonmember_score_mean
            if not math.isnan(member_score_mean) and not math.isnan(nonmember_score_mean)
            else float("nan")
        )
        metric_rows.append(
            {
                "config_id": config_id,
                "measurement": measurement,
                "patient_index": patient_index,
                "class_label": int(rows[0]["class_label"]),
                "appearances": len(rows),
                "member_appearances": len(member_rows),
                "nonmember_appearances": len(nonmember_rows),
                "true_positives_when_member": tp,
                "false_negatives_when_member": fn,
                "true_negatives_when_nonmember": tn,
                "false_positives_when_nonmember": fp,
                "member_correct_rate": tp / len(member_rows) if member_rows else float("nan"),
                "nonmember_correct_rate": tn / len(nonmember_rows) if nonmember_rows else float("nan"),
                "nonmember_false_positive_rate": fp / len(nonmember_rows) if nonmember_rows else float("nan"),
                "threshold_accuracy": (tp + tn) / max(len(rows), 1),
                "f1": patient_f1(tp, fp, fn),
                "auc": patient_auc(member_scores, nonmember_scores),
                "has_both_membership_states": 1 if member_rows and nonmember_rows else 0,
                "score_mean": safe_mean(all_scores),
                "score_std": safe_sample_std(all_scores),
                "member_score_mean": member_score_mean,
                "nonmember_score_mean": nonmember_score_mean,
                "score_gap": score_gap,
                "auc_contribution_mean": safe_mean(all_auc_parts),
                "member_auc_contribution_mean": safe_mean(member_auc_parts),
                "nonmember_auc_contribution_mean": safe_mean(nonmember_auc_parts),
            }
        )
    return metric_rows


def patient_row_is_eligible(row: Dict[str, float], min_state_appearances: int) -> bool:
    try:
        return (
            int(row["member_appearances"]) >= min_state_appearances
            and int(row["nonmember_appearances"]) >= min_state_appearances
            and not math.isnan(float(row["auc"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def summarize_patient_vulnerability(
    patient_rows: Sequence[Dict[str, float]],
    min_state_appearances: int,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for measurement in ("fedmia_i_loss", "fedmia_ii_cosine"):
        prefix = f"{measurement}_patient"
        rows = [row for row in patient_rows if row["measurement"] == measurement]
        both_rows = [row for row in rows if int(row["has_both_membership_states"]) == 1]
        eligible_rows = [
            row for row in rows if patient_row_is_eligible(row, min_state_appearances)
        ]
        auc_values = [float(row["auc"]) for row in eligible_rows]
        gap_values = [float(row["score_gap"]) for row in eligible_rows]
        auc_mean, auc_std = mean_std(auc_values)
        gap_mean, gap_std = mean_std(gap_values)
        summary[f"{prefix}_patients_observed"] = float(len(rows))
        summary[f"{prefix}_patients_with_both_states"] = float(len(both_rows))
        summary[f"{prefix}_eligible_patients"] = float(len(eligible_rows))
        summary[f"{prefix}_auc_mean"] = auc_mean
        summary[f"{prefix}_auc_std"] = auc_std
        summary[f"{prefix}_score_gap_mean"] = gap_mean
        summary[f"{prefix}_score_gap_std"] = gap_std

        if eligible_rows:
            ranked = sorted(
                eligible_rows,
                key=lambda row: (
                    float(row["auc"]),
                    float(row["score_gap"]) if not math.isnan(float(row["score_gap"])) else -999.0,
                ),
            )
            least = ranked[0]
            most = ranked[-1]
            for label, row in (("least", least), ("most", most)):
                summary[f"{prefix}_{label}_vulnerable_patient"] = float(row["patient_index"])
                summary[f"{prefix}_{label}_vulnerable_class_label"] = float(row["class_label"])
                summary[f"{prefix}_{label}_vulnerable_auc"] = float(row["auc"])
                summary[f"{prefix}_{label}_vulnerable_score_gap"] = float(row["score_gap"])
                summary[f"{prefix}_{label}_vulnerable_member_n"] = float(row["member_appearances"])
                summary[f"{prefix}_{label}_vulnerable_nonmember_n"] = float(row["nonmember_appearances"])
        else:
            for label in ("least", "most"):
                summary[f"{prefix}_{label}_vulnerable_patient"] = float("nan")
                summary[f"{prefix}_{label}_vulnerable_class_label"] = float("nan")
                summary[f"{prefix}_{label}_vulnerable_auc"] = float("nan")
                summary[f"{prefix}_{label}_vulnerable_score_gap"] = float("nan")
                summary[f"{prefix}_{label}_vulnerable_member_n"] = 0.0
                summary[f"{prefix}_{label}_vulnerable_nonmember_n"] = 0.0
    return summary


def format_patient_value(value) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_patient_report(
    path: str,
    experiment_id: str,
    observation_csv_path: str,
    patient_csv_path: str,
    patient_rows: Sequence[Dict[str, float]],
    min_state_appearances: int,
):
    lines = [
        "FedMIA BINN Patient-Level Metrics Report",
        f"experiment_id: {experiment_id}",
        f"patient_observation_csv_file: {os.path.relpath(observation_csv_path, REPO_ROOT)}",
        f"patient_metrics_csv_file: {os.path.relpath(patient_csv_path, REPO_ROOT)}",
        "",
        "Interpretation",
        "  auc is defined only when the same patient appears as both IN and OUT across runs.",
        "  auc_contribution_mean is available for every attacked patient and measures that patient's",
        "  pairwise contribution to the global AUC in the runs where it appeared.",
        "  member_correct_rate is the rate of correct IN predictions when the patient was a member.",
        "  nonmember_correct_rate is the rate of correct OUT predictions when the patient was a nonmember.",
        f"  vulnerability rankings require member_n >= {min_state_appearances} and "
        f"nonmember_n >= {min_state_appearances}.",
        "",
    ]
    for measurement in ("fedmia_i_loss", "fedmia_ii_cosine"):
        rows = [row for row in patient_rows if row["measurement"] == measurement]
        both_rows = [row for row in rows if int(row["has_both_membership_states"]) == 1]
        lines.extend(
            [
                f"{measurement}",
                f"  patients_observed: {len(rows)}",
                f"  patients_with_both_in_and_out_states: {len(both_rows)}",
                (
                    "  patients_eligible_for_vulnerability_ranking: "
                    f"{sum(1 for row in rows if patient_row_is_eligible(row, min_state_appearances))}"
                ),
                "  Top member leakage cases",
            ]
        )
        member_rows = [
            row for row in rows if int(row["member_appearances"]) > 0
        ]
        member_rows.sort(
            key=lambda row: (
                -float(row["member_correct_rate"]) if not math.isnan(float(row["member_correct_rate"])) else 1.0,
                -float(row["member_auc_contribution_mean"]) if not math.isnan(float(row["member_auc_contribution_mean"])) else 1.0,
                -float(row["member_score_mean"]) if not math.isnan(float(row["member_score_mean"])) else 1.0,
            )
        )
        for row in member_rows[:10]:
            lines.append(
                "    "
                f"config={row['config_id']} patient={row['patient_index']} class_label={row['class_label']} "
                f"member_n={row['member_appearances']} "
                f"member_correct_rate={format_patient_value(row['member_correct_rate'])} "
                f"member_auc_contribution={format_patient_value(row['member_auc_contribution_mean'])} "
                f"member_score_mean={format_patient_value(row['member_score_mean'])} "
                f"f1={format_patient_value(row['f1'])}"
            )
        lines.append("  Top nonmember false-positive cases")
        nonmember_rows = [
            row for row in rows if int(row["nonmember_appearances"]) > 0
        ]
        nonmember_rows.sort(
            key=lambda row: (
                -float(row["nonmember_false_positive_rate"])
                if not math.isnan(float(row["nonmember_false_positive_rate"]))
                else 1.0,
                -float(row["nonmember_score_mean"]) if not math.isnan(float(row["nonmember_score_mean"])) else 1.0,
            )
        )
        for row in nonmember_rows[:10]:
            lines.append(
                "    "
                f"config={row['config_id']} patient={row['patient_index']} class_label={row['class_label']} "
                f"nonmember_n={row['nonmember_appearances']} "
                f"false_positive_rate={format_patient_value(row['nonmember_false_positive_rate'])} "
                f"nonmember_correct_rate={format_patient_value(row['nonmember_correct_rate'])} "
                f"nonmember_auc_contribution={format_patient_value(row['nonmember_auc_contribution_mean'])} "
                f"nonmember_score_mean={format_patient_value(row['nonmember_score_mean'])}"
            )
        lines.append("  Patients with per-patient AUC available")
        eligible_rows = [
            row for row in both_rows if patient_row_is_eligible(row, min_state_appearances)
        ]
        eligible_rows.sort(
            key=lambda row: (
                -float(row["auc"]),
                -float(row["score_gap"]) if not math.isnan(float(row["score_gap"])) else 1.0,
            )
        )
        for row in eligible_rows[:10]:
            lines.append(
                "    "
                f"config={row['config_id']} patient={row['patient_index']} class_label={row['class_label']} "
                f"auc={format_patient_value(row['auc'])} "
                f"score_gap={format_patient_value(row['score_gap'])} "
                f"f1={format_patient_value(row['f1'])} "
                f"member_n={row['member_appearances']} "
                f"nonmember_n={row['nonmember_appearances']}"
            )
        lines.append("  Least vulnerable patients with per-patient AUC available")
        for row in list(reversed(eligible_rows[-10:])):
            lines.append(
                "    "
                f"config={row['config_id']} patient={row['patient_index']} class_label={row['class_label']} "
                f"auc={format_patient_value(row['auc'])} "
                f"score_gap={format_patient_value(row['score_gap'])} "
                f"f1={format_patient_value(row['f1'])} "
                f"member_n={row['member_appearances']} "
                f"nonmember_n={row['nonmember_appearances']}"
            )
        lines.append("")
    rendered = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rendered + "\n")


def summarize_config(results: Sequence[TrajectoryResult], args) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for key in [
        "train_acc_initial",
        "train_acc_final",
        "holdout_acc_initial",
        "holdout_acc_final",
        "elapsed_seconds",
        "member_count",
        "nonmember_count",
    ]:
        mean, std = mean_std([float(getattr(result, key)) for result in results])
        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std
    metric_keys = sorted(results[0].metrics.keys()) if results else []
    for key in metric_keys:
        mean, std = mean_std([result.metrics[key] for result in results])
        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std
    for measurement in ("fedmia_i_loss", "fedmia_ii_cosine"):
        pooled = pooled_metrics(results, measurement, args)
        for key, value in pooled.items():
            row[f"{measurement}_pooled_{key}"] = value
    return row


def make_grid(args, train_count: int) -> List[GridConfig]:
    client_grid = parse_int_grid(args.client_grid)
    round_grid = parse_int_grid(args.round_grid)
    local_epoch_grid = parse_int_grid(args.local_epoch_grid)
    beta_grid = parse_beta_grid(args.beta_grid)
    sample_fractions = parse_float_grid(args.sample_fraction_grid)
    explicit_counts = None
    if args.samples_per_client_grid.strip().lower() != "auto":
        explicit_counts = parse_int_grid(args.samples_per_client_grid)

    configs: List[GridConfig] = []
    config_id = 1
    for clients in client_grid:
        max_per_client = train_count // clients
        if max_per_client < 1:
            LOGGER.warning("Skipping clients=%s because train_count=%s is too small.", clients, train_count)
            continue
        if explicit_counts is None:
            sample_counts = sorted(
                {
                    max(1, min(max_per_client, int(round(max_per_client * fraction))))
                    for fraction in sample_fractions
                }
            )
            count_to_fraction = {
                count: min(1.0, count / max(max_per_client, 1)) for count in sample_counts
            }
        else:
            sample_counts = [count for count in explicit_counts if count <= max_per_client]
            skipped = [count for count in explicit_counts if count > max_per_client]
            for count in skipped:
                LOGGER.warning(
                    "Skipping samples_per_client=%s for clients=%s; BINN max disjoint count is %s.",
                    count,
                    clients,
                    max_per_client,
                )
            count_to_fraction = {count: None for count in sample_counts}
        for samples_per_client in sample_counts:
            for rounds in round_grid:
                for local_epochs in local_epoch_grid:
                    for beta_label, beta in beta_grid:
                        configs.append(
                            GridConfig(
                                config_id=config_id,
                                clients=clients,
                                rounds=rounds,
                                local_epochs=local_epochs,
                                beta=beta,
                                beta_label=beta_label,
                                sample_fraction=count_to_fraction[samples_per_client],
                                samples_per_client=samples_per_client,
                            )
                        )
                        config_id += 1
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]
    return configs


def write_csv(path: str, rows: Sequence[Dict]):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: str,
    experiment_id: str,
    log_path: str,
    jsonl_path: str,
    csv_path: str,
    patient_observation_csv_path: str,
    patient_metrics_csv_path: str,
    patient_report_path: str,
    args,
    data_summary: Dict,
    config_rows: Sequence[Dict],
    total_elapsed: float,
):
    lines = [
        "FedMIA BINN Paper-Style Grid Report",
        f"experiment_id: {experiment_id}",
        "script: experiments/fedmia_binn_paper_grid.py",
        f"log_file: {os.path.relpath(log_path, REPO_ROOT)}",
        f"jsonl_log_file: {os.path.relpath(jsonl_path, REPO_ROOT)}",
        f"csv_file: {os.path.relpath(csv_path, REPO_ROOT)}",
        f"patient_observation_csv_file: {os.path.relpath(patient_observation_csv_path, REPO_ROOT)}",
        f"patient_metrics_csv_file: {os.path.relpath(patient_metrics_csv_path, REPO_ROOT)}",
        f"patient_metrics_report_file: {os.path.relpath(patient_report_path, REPO_ROOT)}",
        "",
        "Protocol",
        "  Baseline follows FedMIA's FL setup, not the earlier LiRA-style shadow protocol.",
        "  Target client: client 0.",
        "  Qout estimate: non-target client updates in the same communication round.",
        "  FedMIA-I: negative model loss measurement.",
        "  FedMIA-II: gradient-cosine measurement from Eq. (7).",
        f"  Nonmember candidate source: {args.nonmember_source}.",
        "  holdout means globally unseen patients.",
        "  other_clients means patients trained by non-target clients.",
        "  target_nonmembers means all patients absent from target client 0: non-target-client plus holdout patients.",
        "  Non-target clients: update references used to estimate Qout.",
        "  Metrics: AUC, F1 at delta, TPR/TNR/FPR, and TPR@low FPR including 0.1%.",
        "",
        "Data And DAG",
        f"  samples_loaded: {data_summary['samples_loaded']}",
        f"  selected_reactome_mapped_features: {data_summary['selected_features']}",
        f"  graph_nodes: {data_summary['graph_nodes']}",
        f"  graph_edges: {data_summary['graph_edges']}",
        f"  train_pool_count_after_holdout: {data_summary['train_count']}",
        f"  holdout_count: {data_summary['holdout_count']}",
        f"  audit_patient_count: {data_summary['audit_patient_count']}",
        f"  positive_rate: {data_summary['positive_rate']:.6f}",
        "",
        "Configuration",
        f"  runs_per_config: {args.runs}",
        f"  client_grid: {args.client_grid}",
        f"  round_grid: {args.round_grid}",
        f"  local_epoch_grid: {args.local_epoch_grid}",
        f"  beta_grid: {args.beta_grid}",
        f"  sample_fraction_grid: {args.sample_fraction_grid}",
        f"  samples_per_client_grid: {args.samples_per_client_grid}",
        f"  holdout_fraction: {args.holdout_fraction}",
        f"  candidate_count: {args.candidate_count}",
        f"  nonmember_source: {args.nonmember_source}",
        f"  audit_patient_count: {args.audit_patient_count}",
        f"  min_patient_state_appearances: {args.min_patient_state_appearances}",
        f"  threshold_delta: {args.threshold}",
        f"  device_requested: {args.device}",
        f"  gpu: {args.gpu}",
        f"  total_elapsed_seconds: {total_elapsed:.2f}",
        "",
        "BINN Sample-Count Translation",
        "  FedMIA's image experiments use 1,000-10,000 samples/client because those datasets are large.",
        "  For BINN prostate data, samples/client means patient records assigned to one FL client.",
        "  With one-tenth holdout, the maximum disjoint samples/client is floor(train_pool / clients).",
        "  The auto mode uses requested fractions of that BINN-specific maximum.",
        "",
        "Grid Results",
    ]
    for row in config_rows:
        lines.extend(
            [
                (
                    f"  config={int(row['config_id']):03d} clients={int(row['clients'])} "
                    f"rounds={int(row['rounds'])} local_epochs={int(row['local_epochs'])} "
                    f"beta={row['beta_label']} samples_per_client={int(row['samples_per_client'])}"
                ),
                (
                    f"    train_acc_final={row['train_acc_final_mean']:.6f} +/- "
                    f"{row['train_acc_final_std']:.6f}"
                ),
                (
                    f"    holdout_acc_final={row['holdout_acc_final_mean']:.6f} +/- "
                    f"{row['holdout_acc_final_std']:.6f}"
                ),
                (
                    f"    FedMIA-I loss auc={row['fedmia_i_loss_auc_mean']:.6f} +/- "
                    f"{row['fedmia_i_loss_auc_std']:.6f} "
                    f"f1={row['fedmia_i_loss_f1_mean']:.6f} +/- {row['fedmia_i_loss_f1_std']:.6f} "
                    f"tpr@fpr0.001={row['fedmia_i_loss_tpr_at_fpr_0.001_mean']:.6f}"
                ),
                (
                    f"    FedMIA-I pooled auc={row['fedmia_i_loss_pooled_auc']:.6f} "
                    f"f1={row['fedmia_i_loss_pooled_f1']:.6f} "
                    f"tpr={row['fedmia_i_loss_pooled_tpr']:.6f} "
                    f"tnr={row['fedmia_i_loss_pooled_tnr']:.6f} "
                    f"fpr={row['fedmia_i_loss_pooled_fpr']:.6f} "
                    f"tpr@fpr0.001={row['fedmia_i_loss_pooled_tpr_at_fpr_0.001']:.6f}"
                ),
                (
                    f"    FedMIA-II cosine auc={row['fedmia_ii_cosine_auc_mean']:.6f} +/- "
                    f"{row['fedmia_ii_cosine_auc_std']:.6f} "
                    f"f1={row['fedmia_ii_cosine_f1_mean']:.6f} +/- {row['fedmia_ii_cosine_f1_std']:.6f} "
                    f"tpr@fpr0.001={row['fedmia_ii_cosine_tpr_at_fpr_0.001_mean']:.6f}"
                ),
                (
                    f"    FedMIA-II pooled auc={row['fedmia_ii_cosine_pooled_auc']:.6f} "
                    f"f1={row['fedmia_ii_cosine_pooled_f1']:.6f} "
                    f"tpr={row['fedmia_ii_cosine_pooled_tpr']:.6f} "
                    f"tnr={row['fedmia_ii_cosine_pooled_tnr']:.6f} "
                    f"fpr={row['fedmia_ii_cosine_pooled_fpr']:.6f} "
                    f"tpr@fpr0.001={row['fedmia_ii_cosine_pooled_tpr_at_fpr_0.001']:.6f}"
                ),
                (
                    "    Patient vulnerability FedMIA-I "
                    f"eligible={int(row.get('fedmia_i_loss_patient_eligible_patients', 0))} "
                    f"auc_mean={row.get('fedmia_i_loss_patient_auc_mean', float('nan')):.6f} "
                    f"most_patient={row.get('fedmia_i_loss_patient_most_vulnerable_patient', float('nan')):.0f} "
                    f"most_auc={row.get('fedmia_i_loss_patient_most_vulnerable_auc', float('nan')):.6f} "
                    f"least_patient={row.get('fedmia_i_loss_patient_least_vulnerable_patient', float('nan')):.0f} "
                    f"least_auc={row.get('fedmia_i_loss_patient_least_vulnerable_auc', float('nan')):.6f}"
                ),
                (
                    "    Patient vulnerability FedMIA-II "
                    f"eligible={int(row.get('fedmia_ii_cosine_patient_eligible_patients', 0))} "
                    f"auc_mean={row.get('fedmia_ii_cosine_patient_auc_mean', float('nan')):.6f} "
                    f"most_patient={row.get('fedmia_ii_cosine_patient_most_vulnerable_patient', float('nan')):.0f} "
                    f"most_auc={row.get('fedmia_ii_cosine_patient_most_vulnerable_auc', float('nan')):.6f} "
                    f"least_patient={row.get('fedmia_ii_cosine_patient_least_vulnerable_patient', float('nan')):.0f} "
                    f"least_auc={row.get('fedmia_ii_cosine_patient_least_vulnerable_auc', float('nan')):.6f}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Commentary",
            "  This report intentionally does not include LiRA-style shadow-model metrics.",
            "  Low-FPR metrics are sensitive to the number of nonmember candidates; pooled rows",
            "  are therefore the most stable low-FPR readout for small BINN grid cells.",
        ]
    )
    rendered = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(rendered)


def main():
    args = parse_args()
    experiment_id = make_experiment_id()
    report_root = args.report_dir
    log_root = args.log_dir
    args.report_dir = os.path.join(report_root, experiment_id)
    args.log_dir = os.path.join(log_root, experiment_id)
    log_path, jsonl_path = configure_logging(args, experiment_id)
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, f"{experiment_id}.txt")
    csv_path = os.path.join(args.report_dir, f"{experiment_id}.csv")
    patient_observation_csv_path = os.path.join(
        args.report_dir, f"{experiment_id}_patient_observations.csv"
    )
    patient_metrics_csv_path = os.path.join(args.report_dir, f"{experiment_id}_patient_metrics.csv")
    patient_report_path = os.path.join(args.report_dir, f"{experiment_id}_patient_metrics.txt")
    start_time = time.time()

    LOGGER.info("experiment_id=%s", experiment_id)
    LOGGER.info("args=%s", vars(args))
    device = resolve_device(args)
    set_seed(args.seed)

    feature_index = research.load_pnet_feature_index()
    graph, selected_features = research.build_real_reactome_feature_graph(
        feature_index,
        feature_limit=args.feature_limit,
    )
    x, y = research.load_pnet_data(selected_features, args)
    split_rng = np.random.default_rng(args.seed)
    train_indices, holdout_indices = stratified_holdout_split(y, args.holdout_fraction, split_rng)
    audit_rng = np.random.default_rng(args.seed + 17_171)
    audit_indices = choose_audit_patient_indices(
        y,
        train_indices,
        args.audit_patient_count,
        audit_rng,
    )
    configs = make_grid(args, len(train_indices))
    data_summary = {
        "samples_loaded": int(x.shape[0]),
        "selected_features": int(x.shape[1]),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "train_count": len(train_indices),
        "holdout_count": len(holdout_indices),
        "audit_patient_count": len(audit_indices),
        "positive_rate": float(y.mean().item()),
    }
    LOGGER.info(
        "loaded data samples=%s features=%s graph_nodes=%s graph_edges=%s train_pool=%s holdout=%s configs=%s",
        data_summary["samples_loaded"],
        data_summary["selected_features"],
        data_summary["graph_nodes"],
        data_summary["graph_edges"],
        data_summary["train_count"],
        data_summary["holdout_count"],
        len(configs),
    )
    append_jsonl(
        jsonl_path,
        {
            "event": "experiment_start",
            "experiment_id": experiment_id,
            "args": vars(args),
            "data_summary": data_summary,
            "audit_patient_indices_head": [int(idx) for idx in audit_indices[:20]],
            "configs": [config.__dict__ for config in configs],
        },
    )
    if not configs:
        raise ValueError("Grid is empty; check client/sample settings.")

    config_rows = []
    all_results: List[TrajectoryResult] = []
    global_run_id = 1
    for config in configs:
        LOGGER.info(
            "config=%03d start clients=%s rounds=%s local_epochs=%s beta=%s samples_per_client=%s",
            config.config_id,
            config.clients,
            config.rounds,
            config.local_epochs,
            config.beta_label,
            config.samples_per_client,
        )
        config_results = []
        for run_offset in range(args.runs):
            run_seed = args.seed + config.config_id * 100_000 + run_offset * 1_009
            result = run_one_trajectory(
                config,
                global_run_id,
                run_seed,
                graph,
                x,
                y,
                train_indices,
                holdout_indices,
                audit_indices,
                args,
                device,
                jsonl_path,
            )
            config_results.append(result)
            all_results.append(result)
            global_run_id += 1
        summary = summarize_config(config_results, args)
        row = {
            "config_id": config.config_id,
            "clients": config.clients,
            "rounds": config.rounds,
            "local_epochs": config.local_epochs,
            "beta_label": config.beta_label,
            "nonmember_source": args.nonmember_source,
            "audit_patient_count": len(audit_indices),
            "sample_fraction": config.sample_fraction if config.sample_fraction is not None else "",
            "samples_per_client": config.samples_per_client,
            **summary,
        }
        config_patient_observation_rows = make_patient_observation_rows(config_results, args, y)
        config_patient_metric_rows = make_patient_metric_rows(config_patient_observation_rows)
        row.update(
            summarize_patient_vulnerability(
                config_patient_metric_rows,
                args.min_patient_state_appearances,
            )
        )
        config_rows.append(row)
        write_csv(csv_path, config_rows)
        append_jsonl(jsonl_path, {"event": "config_result", **row})
        LOGGER.info(
            "config=%03d result fedmia_i_auc=%.4f fedmia_ii_auc=%.4f holdout_acc=%.4f",
            config.config_id,
            row["fedmia_i_loss_auc_mean"],
            row["fedmia_ii_cosine_auc_mean"],
            row["holdout_acc_final_mean"],
        )

    total_elapsed = time.time() - start_time
    patient_observation_rows = make_patient_observation_rows(all_results, args, y)
    patient_metric_rows = make_patient_metric_rows(patient_observation_rows)
    write_csv(patient_observation_csv_path, patient_observation_rows)
    write_csv(patient_metrics_csv_path, patient_metric_rows)
    write_patient_report(
        patient_report_path,
        experiment_id,
        patient_observation_csv_path,
        patient_metrics_csv_path,
        patient_metric_rows,
        args.min_patient_state_appearances,
    )
    write_report(
        report_path,
        experiment_id,
        log_path,
        jsonl_path,
        csv_path,
        patient_observation_csv_path,
        patient_metrics_csv_path,
        patient_report_path,
        args,
        data_summary,
        config_rows,
        total_elapsed,
    )
    LOGGER.info("report_path=%s", report_path)
    LOGGER.info("patient_metrics_report_path=%s", patient_report_path)


if __name__ == "__main__":
    main()
