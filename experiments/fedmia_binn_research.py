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
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import (  # noqa: E402
    FedMIAConfig,
    FedMIARoundMeasurements,
    cosine_measurement,
    evaluate_membership,
)
from models.binn import binn  # noqa: E402
from utils.federated import fed_avg_state_dicts  # noqa: E402


LOGGER = logging.getLogger("fedmia_binn_research")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
PROSTATE_DIR = os.path.join(REPO_ROOT, "data", "datasets", "ProstateCancer")
REACTOME_DIR = os.path.join(REPO_ROOT, "data", "datasets", "Reactome")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
LOG_DIR = os.path.join(REPO_ROOT, "logs")


@dataclass
class SplitBundle:
    client_indices: List[List[int]]
    holdout_indices: List[int]
    member_indices: List[int]
    nonmember_indices: List[int]


@dataclass
class RunResult:
    run_id: int
    seed: int
    metrics: Dict[str, float]
    initial_acc: float
    final_acc: float
    elapsed_seconds: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Research-grade FedMIA evaluation on BINN with a real Reactome DAG."
    )
    parser.add_argument("--runs", type=int, default=200, help="number of independent FL trajectories")
    parser.add_argument("--rounds", type=int, default=20, help="communication rounds per FL trajectory")
    parser.add_argument("--local-epochs", type=int, default=2, help="local training epochs per round")
    parser.add_argument("--num-clients", type=int, default=5, help="federated clients per trajectory")
    parser.add_argument("--samples-per-client", type=int, default=64, help="patient samples per client")
    parser.add_argument("--candidate-count", type=int, default=32, help="member and non-member candidates")
    parser.add_argument("--batch-size", type=int, default=16, help="local training batch size")
    parser.add_argument("--lr", type=float, default=0.03, help="local optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="local optimizer weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--threshold", type=float, default=0.5, help="FedMIA delta threshold")
    parser.add_argument(
        "--threshold-grid",
        default="0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="comma-separated deltas used to report threshold-sweep F1 diagnostics",
    )
    parser.add_argument("--outlier-std-factor", type=float, default=3.0, help="FedMIA 3-sigma filter")
    parser.add_argument("--min-variance", type=float, default=1e-8, help="FedMIA Gaussian variance floor")
    parser.add_argument("--seed", type=int, default=20260509, help="base random seed")
    parser.add_argument(
        "--resample-splits",
        action="store_true",
        help="vary patient/client splits per run instead of holding the same split fixed",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="training device")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value when using CUDA")
    parser.add_argument("--x-file", default="pnet_x.npy", help="ProstateCancer feature array")
    parser.add_argument("--y-file", default="pnet_y.npy", help="ProstateCancer label array")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="optional sample cap for debugging; 0 means use all available samples",
    )
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=0,
        help="optional pnet feature-column cap for debugging; 0 means all Reactome-mapped features",
    )
    parser.add_argument("--report-dir", default=REPORT_DIR, help="directory for final reports")
    parser.add_argument("--log-dir", default=LOG_DIR, help="directory for execution logs")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_experiment_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"fedmia_binn_research_{stamp}_{uuid.uuid4().hex[:8]}"


def configure_logging(args, experiment_id: str) -> Tuple[str, str]:
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{experiment_id}.log")
    jsonl_path = os.path.join(args.log_dir, f"{experiment_id}.jsonl")
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path, jsonl_path


def append_jsonl(path: str, event: Dict):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def resolve_device(args):
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
    return torch.device("cpu")


def load_pnet_feature_index() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROSTATE_DIR, "pnet_index.csv"))


def read_human_reactome_base_graph() -> nx.DiGraph:
    pathway_info = pd.read_csv(
        os.path.join(REACTOME_DIR, "ReactomePathways.txt"),
        sep="\t",
        header=None,
        names=["PathwayID", "PathwayName", "Species"],
    )
    human_pathways = set(pathway_info.loc[pathway_info["Species"] == "Homo sapiens", "PathwayID"])
    pathway_relations = pd.read_csv(
        os.path.join(REACTOME_DIR, "ReactomePathwaysRelation.txt"),
        sep="\t",
        header=None,
        names=["Parent", "Child"],
    )
    pathway_relations = pathway_relations[
        pathway_relations["Parent"].isin(human_pathways)
        & pathway_relations["Child"].isin(human_pathways)
    ]
    return nx.from_pandas_edgelist(
        pathway_relations,
        source="Child",
        target="Parent",
        create_using=nx.DiGraph,
    )


def read_human_uniprot_reactome_annotations(pnet_proteins: Iterable[str], reactome_terms: Iterable[str]):
    annotations = pd.read_csv(
        os.path.join(REACTOME_DIR, "UniProt2Reactome.txt"),
        sep="\t",
        header=None,
        names=["UniprotID", "PathwayID", "PathwayURL", "PathwayName", "Evidence", "Species"],
    )
    return annotations[
        (annotations["Species"] == "Homo sapiens")
        & (annotations["UniprotID"].isin(set(pnet_proteins)))
        & (annotations["PathwayID"].isin(set(reactome_terms)))
    ][["UniprotID", "PathwayID"]].drop_duplicates()


def build_real_reactome_feature_graph(feature_index: pd.DataFrame, feature_limit: int = 0):
    """
    Convert pnet columns into exact BINN input nodes over the real Reactome DAG.

    The prostate matrix has one column per (UniProt, feature type), while Reactome
    is protein/pathway-level. We keep the real Reactome pathway hierarchy and add
    deterministic leaf nodes:

        feature column -> UniProt adapter -> Reactome pathway -> broader pathway

    This preserves exact input-column alignment without pretending that mutation
    and CNV are separate Reactome proteins.
    """
    feature_index = feature_index.copy()
    feature_index["UniprotID"] = feature_index["UniprotID"].astype(str)
    feature_index["Type"] = feature_index["Type"].astype(str)
    base_graph = read_human_reactome_base_graph()
    annotations = read_human_uniprot_reactome_annotations(
        feature_index["UniprotID"].unique(),
        base_graph.nodes,
    )
    annotated_proteins = set(annotations["UniprotID"])

    selected = feature_index[feature_index["UniprotID"].isin(annotated_proteins)].copy()
    selected["column_index"] = selected.index
    if feature_limit and feature_limit > 0:
        selected = selected.iloc[:feature_limit].copy()
        annotations = annotations[annotations["UniprotID"].isin(set(selected["UniprotID"]))]

    selected["feature_node"] = [
        f"feature::{row.UniprotID}::{row.Type}::{row.column_index}"
        for row in selected.itertuples()
    ]
    selected["protein_node"] = [f"protein::{value}" for value in selected["UniprotID"]]

    reactome_terms = set(annotations["PathwayID"])
    for pathway_id in list(annotations["PathwayID"].unique()):
        reactome_terms.update(nx.descendants(base_graph, pathway_id))
    reactome_subgraph = base_graph.subgraph(reactome_terms).copy()

    graph = nx.MultiDiGraph()
    graph.name = "ReactomeFeatureBINN"
    feature_nodes = selected["feature_node"].tolist()
    protein_nodes = sorted(set(selected["protein_node"]))
    term_nodes = list(nx.topological_sort(reactome_subgraph))

    graph.add_nodes_from(feature_nodes, type="Feature")
    graph.add_nodes_from(protein_nodes, type="Protein")
    graph.add_nodes_from(term_nodes, type="Term")

    for row in selected.itertuples():
        graph.add_edge(row.feature_node, row.protein_node, key="Feature_Protein")

    protein_node_by_uniprot = {
        uniprot: f"protein::{uniprot}"
        for uniprot in selected["UniprotID"].unique()
    }
    for row in annotations.itertuples():
        protein_node = protein_node_by_uniprot.get(row.UniprotID)
        if protein_node is not None and row.PathwayID in graph:
            graph.add_edge(protein_node, row.PathwayID, key="Protein_Reactome")

    for child, parent in reactome_subgraph.edges():
        graph.add_edge(child, parent, key="Reactome_Relation")

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Reactome feature graph must be a DAG.")
    if [node for node in graph.nodes if graph.in_degree(node) == 0] != feature_nodes:
        raise ValueError("Graph input order does not match selected pnet feature order.")
    return graph, selected


def load_pnet_data(selected_features: pd.DataFrame, args) -> Tuple[torch.Tensor, torch.Tensor]:
    x_path = os.path.join(PROSTATE_DIR, args.x_file)
    y_path = os.path.join(PROSTATE_DIR, args.y_file)
    x = np.load(x_path).astype("float32", copy=False)
    y = np.load(y_path).astype("float32", copy=False).reshape(-1, 1)
    if args.max_samples and args.max_samples > 0:
        x = x[: args.max_samples]
        y = y[: args.max_samples]
    x = x[:, selected_features["column_index"].to_numpy()].copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def balanced_indices(labels: torch.Tensor, count: int, rng: np.random.Generator, exclude=None):
    exclude = set(exclude or [])
    label_values = labels.view(-1).cpu().numpy()
    selected = []
    per_class = count // 2
    for label in (0.0, 1.0):
        candidates = [idx for idx, value in enumerate(label_values) if value == label and idx not in exclude]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    if len(selected) < count:
        remaining = [idx for idx in range(len(label_values)) if idx not in exclude and idx not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected[:count]


def make_split(labels: torch.Tensor, args, seed: int) -> SplitBundle:
    rng = np.random.default_rng(seed)
    used = set()
    client_indices = []
    for _ in range(args.num_clients):
        indices = balanced_indices(labels, args.samples_per_client, rng, used)
        used.update(indices)
        client_indices.append(indices)
    holdout_indices = balanced_indices(labels, args.candidate_count, rng, used)
    member_indices = client_indices[0][: args.candidate_count]
    nonmember_indices = holdout_indices[: args.candidate_count]
    return SplitBundle(client_indices, holdout_indices, member_indices, nonmember_indices)


def binary_loss(logits, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.view_as(logits))


def model_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits).view(-1) >= 0.5).float()
            truth = y.view(-1).float()
            correct += pred.eq(truth).sum().item()
            total += truth.numel()
    return correct / max(total, 1)


def train_one_client(model, loader, device, args, run_id: int, round_id: int, client_id: int, jsonl_path: str):
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    epoch_losses = []
    for local_epoch in range(1, args.local_epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = binary_loss(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            pred = (torch.sigmoid(logits).view(-1) >= 0.5).float()
            truth = y.view(-1).float()
            total_correct += pred.eq(truth).sum().item()
            total += truth.numel()
        avg_loss = total_loss / max(total, 1)
        avg_acc = total_correct / max(total, 1)
        epoch_losses.append(avg_loss)
        LOGGER.info(
            "run=%03d round=%03d client=%02d local_epoch=%02d loss=%.6f acc=%.4f",
            run_id,
            round_id,
            client_id,
            local_epoch,
            avg_loss,
            avg_acc,
        )
        append_jsonl(
            jsonl_path,
            {
                "event": "local_epoch",
                "run": run_id,
                "round": round_id,
                "client": client_id,
                "local_epoch": local_epoch,
                "loss": avg_loss,
                "accuracy": avg_acc,
            },
        )
    return copy.deepcopy(model.state_dict()), float(np.mean(epoch_losses))


def state_update(global_state, local_state, parameter_names):
    return {
        key: (global_state[key].detach().cpu() - local_state[key].detach().cpu())
        for key in parameter_names
    }


def sample_gradient(global_model, sample, device):
    x, y = sample
    global_model.zero_grad(set_to_none=True)
    global_model.eval()
    logits = global_model(x.unsqueeze(0).to(device))
    loss = binary_loss(logits, y.view(1, 1).to(device))
    params = [(name, param) for name, param in global_model.named_parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, [param for _, param in params], allow_unused=True)
    return {
        name: (torch.zeros_like(param).detach().cpu() if grad is None else grad.detach().cpu())
        for (name, param), grad in zip(params, grads)
    }


def measure_round(global_model, target_update, reference_updates, samples, device, round_id, split):
    gradients = [sample_gradient(global_model, sample, device) for sample in samples]
    target_measurements = [cosine_measurement(target_update, gradient) for gradient in gradients]
    reference_measurements = [
        [cosine_measurement(reference_update, gradient) for gradient in gradients]
        for reference_update in reference_updates
    ]
    return FedMIARoundMeasurements(
        target_measurements=target_measurements,
        reference_measurements=reference_measurements,
        round_id=round_id,
        metadata={"split": split, "measurement": "gradient_cosine"},
    )


def confusion_metrics(evaluation) -> Dict[str, float]:
    return confusion_metrics_from_predictions(
        evaluation.member_scores.predictions,
        evaluation.nonmember_scores.predictions,
    )


def confusion_metrics_from_predictions(
    member_predictions: Sequence[int],
    nonmember_predictions: Sequence[int],
) -> Dict[str, float]:
    tp = sum(member_predictions)
    fn = len(member_predictions) - tp
    fp = sum(nonmember_predictions)
    tn = len(nonmember_predictions) - fp
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "threshold_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }


def parse_threshold_grid(value: str) -> List[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if part:
            thresholds.append(float(part))
    return sorted(set(thresholds))


def metrics_at_threshold(evaluation, threshold: float) -> Dict[str, float]:
    member_predictions = [
        1 if score > threshold else 0
        for score in evaluation.member_scores.aggregate_scores
    ]
    nonmember_predictions = [
        1 if score > threshold else 0
        for score in evaluation.nonmember_scores.aggregate_scores
    ]
    metrics = confusion_metrics_from_predictions(member_predictions, nonmember_predictions)
    metrics["threshold"] = threshold
    return metrics


def threshold_sweep(evaluation, thresholds: Sequence[float]) -> List[Dict[str, float]]:
    rows = [metrics_at_threshold(evaluation, threshold) for threshold in thresholds]
    rows.sort(key=lambda row: (row["f1"], row["tnr"], row["tpr"]), reverse=True)
    return rows


def run_one_trajectory(
    run_id: int,
    run_seed: int,
    graph,
    x,
    y,
    split: SplitBundle,
    args,
    device,
    jsonl_path: str,
):
    run_start = time.time()
    set_seed(run_seed)
    generator = torch.Generator().manual_seed(run_seed)
    client_datasets = [TensorDataset(x[indices], y[indices]) for indices in split.client_indices]
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
    member_samples = [(x[idx], y[idx]) for idx in split.member_indices]
    nonmember_samples = [(x[idx], y[idx]) for idx in split.nonmember_indices]

    global_model = binn(graph=graph, output_size=1, dropout_prob=0.0, output_last_layers=1).to(device)
    initial_acc = model_accuracy(global_model, combined_loader, device)
    member_rounds = []
    nonmember_rounds = []

    LOGGER.info("run=%03d started seed=%s initial_acc=%.4f", run_id, run_seed, initial_acc)
    for round_id in range(1, args.rounds + 1):
        round_start = time.time()
        global_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        local_updates = []
        local_losses = []
        start_acc = model_accuracy(global_model, combined_loader, device)
        LOGGER.info("run=%03d round=%03d start global_acc=%.4f", run_id, round_id, start_acc)

        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_state, local_loss = train_one_client(
                local_model,
                loader,
                device,
                args,
                run_id,
                round_id,
                client_id,
                jsonl_path,
            )
            local_states.append(local_state)
            local_losses.append(local_loss)
            parameter_names = [name for name, _ in local_model.named_parameters()]
            local_updates.append(state_update(global_state, local_state, parameter_names))
            del local_model

        member_rounds.append(
            measure_round(
                global_model,
                target_update=local_updates[0],
                reference_updates=local_updates[1:],
                samples=member_samples,
                device=device,
                round_id=round_id,
                split="member",
            )
        )
        nonmember_rounds.append(
            measure_round(
                global_model,
                target_update=local_updates[0],
                reference_updates=local_updates[1:],
                samples=nonmember_samples,
                device=device,
                round_id=round_id,
                split="nonmember",
            )
        )

        averaged_state = fed_avg_state_dicts(local_states, [len(dataset) for dataset in client_datasets])
        global_model.load_state_dict(averaged_state)
        end_acc = model_accuracy(global_model, combined_loader, device)
        round_elapsed = time.time() - round_start
        LOGGER.info(
            "run=%03d round=%03d end global_acc=%.4f mean_client_loss=%.6f elapsed=%.2fs",
            run_id,
            round_id,
            end_acc,
            float(np.mean(local_losses)),
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
                "elapsed_seconds": round_elapsed,
            },
        )

    final_acc = model_accuracy(global_model, combined_loader, device)
    evaluation = evaluate_membership(
        member_rounds,
        nonmember_rounds,
        config=FedMIAConfig(
            threshold=args.threshold,
            outlier_std_factor=args.outlier_std_factor,
            min_variance=args.min_variance,
        ),
    )
    metrics = confusion_metrics(evaluation)
    sweep_rows = threshold_sweep(evaluation, parse_threshold_grid(args.threshold_grid))
    best_threshold_metrics = sweep_rows[0] if sweep_rows else {
        "threshold": args.threshold,
        "f1": metrics["f1"],
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
    }
    metrics.update(
        {
            "auc": evaluation.auc,
            "log_auc": evaluation.log_auc,
            "tpr_at_fpr_0.1": evaluation.tprs["0.1"],
            "tpr_at_fpr_0.01": evaluation.tprs["0.01"],
            "member_score_mean": float(np.mean(evaluation.member_scores.aggregate_scores)),
            "nonmember_score_mean": float(np.mean(evaluation.nonmember_scores.aggregate_scores)),
            "best_f1_threshold": best_threshold_metrics["threshold"],
            "best_f1": best_threshold_metrics["f1"],
            "best_f1_tpr": best_threshold_metrics["tpr"],
            "best_f1_tnr": best_threshold_metrics["tnr"],
        }
    )
    elapsed = time.time() - run_start
    LOGGER.info(
        "run=%03d finished final_acc=%.4f auc=%.4f f1=%.4f tpr=%.4f tnr=%.4f "
        "best_f1=%.4f best_threshold=%.2f elapsed=%.2fs",
        run_id,
        final_acc,
        metrics["auc"],
        metrics["f1"],
        metrics["tpr"],
        metrics["tnr"],
        metrics["best_f1"],
        metrics["best_f1_threshold"],
        elapsed,
    )
    append_jsonl(
        jsonl_path,
        {
            "event": "run_result",
            "run": run_id,
            "seed": run_seed,
            "initial_acc": initial_acc,
            "final_acc": final_acc,
            "elapsed_seconds": elapsed,
            "threshold_sweep": sweep_rows,
            **metrics,
        },
    )
    return RunResult(run_id, run_seed, metrics, initial_acc, final_acc, elapsed)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1 if len(arr) > 1 else 0))


def build_report(
    experiment_id: str,
    args,
    graph,
    selected_features,
    results: Sequence[RunResult],
    log_path: str,
    jsonl_path: str,
    total_elapsed: float,
) -> str:
    node_types = nx.get_node_attributes(graph, "type")
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
        "member_score_mean",
        "nonmember_score_mean",
        "best_f1",
        "best_f1_threshold",
        "best_f1_tpr",
        "best_f1_tnr",
    ]
    lines = [
        "FedMIA BINN Research Evaluation Report",
        f"experiment_id: {experiment_id}",
        f"script: experiments/fedmia_binn_research.py",
        f"log_file: {log_path}",
        f"jsonl_log_file: {jsonl_path}",
        "",
        "Protocol",
        "  Unit of repetition: one full federated BINN training trajectory.",
        "  Membership evidence: target/non-target client updates across communication rounds.",
        "  Aggregation: metric mean +/- sample standard deviation across trajectories.",
        f"  fixed_patient_split: {not args.resample_splits}",
        "",
        "Data",
        f"  x_file: data/datasets/ProstateCancer/{args.x_file}",
        f"  y_file: data/datasets/ProstateCancer/{args.y_file}",
        f"  samples_loaded: {args.max_samples if args.max_samples else 'all'}",
        f"  selected_reactome_mapped_features: {len(selected_features)}",
        "",
        "Real DAG Summary",
        f"  graph_name: {graph.name}",
        f"  nodes: {graph.number_of_nodes()}",
        f"  edges: {graph.number_of_edges()}",
        f"  feature_input_nodes: {sum(1 for value in node_types.values() if value == 'Feature')}",
        f"  protein_adapter_nodes: {sum(1 for value in node_types.values() if value == 'Protein')}",
        f"  reactome_term_nodes: {sum(1 for value in node_types.values() if value == 'Term')}",
        "",
        "Training Configuration",
        f"  runs: {args.runs}",
        f"  clients: {args.num_clients}",
        f"  rounds: {args.rounds}",
        f"  local_epochs: {args.local_epochs}",
        f"  samples_per_client: {args.samples_per_client}",
        f"  candidate_count_each_class: {args.candidate_count}",
        f"  batch_size: {args.batch_size}",
        f"  lr: {args.lr}",
        f"  device_requested: {args.device}",
        f"  gpu: {args.gpu}",
        f"  threshold_delta: {args.threshold}",
        f"  threshold_grid: {args.threshold_grid}",
        f"  total_elapsed_seconds: {total_elapsed:.2f}",
        "",
        "Aggregated Performance Metrics",
    ]
    for name in metric_names:
        mean, std = mean_std([result.metrics[name] for result in results])
        lines.append(f"  {name}: {mean:.6f} +/- {std:.6f}")

    init_mean, init_std = mean_std([result.initial_acc for result in results])
    final_mean, final_std = mean_std([result.final_acc for result in results])
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
    for result in results:
        lines.append(
            f"  run={result.run_id:03d} seed={result.seed} "
            f"auc={result.metrics['auc']:.6f} f1={result.metrics['f1']:.6f} "
            f"tpr={result.metrics['tpr']:.6f} tnr={result.metrics['tnr']:.6f} "
            f"final_acc={result.final_acc:.6f} elapsed={result.elapsed_seconds:.2f}s"
        )

    if final_mean < 0.65:
        comment = (
            "Model training is weak under this configuration, so FedMIA metrics should be "
            "treated as an execution check rather than a privacy-risk estimate. Increase "
            "rounds/local epochs or tune optimization before comparing with central BINN."
        )
    else:
        comment = (
            "Model training reached a usable regime. Compare FedMIA's mean +/- std metrics "
            "with the central BINN/LiRA table while noting the repetition unit is an FL trajectory."
        )
    lines.extend(["", "Commentary", f"  {comment}"])
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

    feature_index = load_pnet_feature_index()
    graph, selected_features = build_real_reactome_feature_graph(feature_index, args.feature_limit)
    x, y = load_pnet_data(selected_features, args)
    LOGGER.info(
        "loaded data samples=%s features=%s positive_rate=%.4f",
        x.shape[0],
        x.shape[1],
        float(y.mean().item()),
    )
    LOGGER.info(
        "real_dag nodes=%s edges=%s selected_features=%s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(selected_features),
    )

    fixed_split = make_split(y, args, args.seed)
    results = []
    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx * 1009
        split = make_split(y, args, run_seed) if args.resample_splits else fixed_split
        result = run_one_trajectory(
            run_idx,
            run_seed,
            graph,
            x,
            y,
            split,
            args,
            device,
            jsonl_path,
        )
        results.append(result)

    total_elapsed = time.time() - total_start
    report_text = build_report(
        experiment_id,
        args,
        graph,
        selected_features,
        results,
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
