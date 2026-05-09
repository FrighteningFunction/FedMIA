import copy
import os
import sys
import time
import uuid
import unittest
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from attacks.fedmia import FedMIAConfig, evaluate_membership
from models.binn import binn
from test_tars_smoke import (
    _accuracy,
    _make_client_splits,
    _mean,
    _measure_round,
    _set_seed,
    _state_update,
    _train_one_client,
)
from utils.federated import fed_avg_state_dicts


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
PROSTATE_DIR = os.path.join(REPO_ROOT, "data", "datasets", "ProstateCancer")
REACTOME_DIR = os.path.join(REPO_ROOT, "data", "datasets", "Reactome")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")


def _load_pnet_feature_index():
    return pd.read_csv(os.path.join(PROSTATE_DIR, "pnet_index.csv"))


def _load_real_pnet_chunk_for_feature_rows(feature_rows, max_samples=64):
    x = np.load(os.path.join(PROSTATE_DIR, "pnet_x_100.npy"))[:max_samples].astype("float32", copy=False)
    y = np.load(os.path.join(PROSTATE_DIR, "pnet_y_100.npy"))[:max_samples].astype("float32", copy=False).reshape(-1, 1)
    x = x[:, feature_rows].copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)

    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def _read_human_reactome_base_graph():
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


def _read_human_uniprot_reactome_annotations(pnet_proteins, reactome_terms):
    annotations = pd.read_csv(
        os.path.join(REACTOME_DIR, "UniProt2Reactome.txt"),
        sep="\t",
        header=None,
        names=["UniprotID", "PathwayID", "PathwayURL", "PathwayName", "Evidence", "Species"],
    )
    return annotations[
        (annotations["Species"] == "Homo sapiens")
        & (annotations["UniprotID"].isin(pnet_proteins))
        & (annotations["PathwayID"].isin(reactome_terms))
    ][["UniprotID", "PathwayID"]].drop_duplicates()


def build_real_reactome_feature_graph(feature_index):
    """
    Build a real Reactome DAG whose input nodes are exactly selected pnet_x columns.

    Reactome annotates proteins, while pnet_x has one column per (UniProt, feature type).
    We therefore keep the real Reactome pathway hierarchy and add deterministic feature
    leaves: feature -> UniProt adapter node -> Reactome pathway -> broader pathway.
    """
    feature_index = feature_index.copy()
    feature_index["UniprotID"] = feature_index["UniprotID"].astype(str)
    feature_index["Type"] = feature_index["Type"].astype(str)

    base_graph = _read_human_reactome_base_graph()
    annotations = _read_human_uniprot_reactome_annotations(
        set(feature_index["UniprotID"]),
        set(base_graph.nodes),
    )
    annotated_proteins = set(annotations["UniprotID"])

    selected = feature_index[feature_index["UniprotID"].isin(annotated_proteins)].copy()
    selected["column_index"] = selected.index
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
        raise ValueError("The real Reactome feature graph must be a DAG.")

    input_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
    if input_nodes != feature_nodes:
        raise ValueError("Reactome feature graph input order does not match pnet feature order.")

    return graph, selected


def _confusion_metrics(evaluation):
    member_predictions = evaluation.member_scores.predictions
    nonmember_predictions = evaluation.nonmember_scores.predictions
    tp = sum(member_predictions)
    fn = len(member_predictions) - tp
    fp = sum(nonmember_predictions)
    tn = len(nonmember_predictions) - fp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }


def _build_report(report_id, graph, selected_features, config, log_lines, evaluation, initial_acc, final_acc, elapsed):
    metrics = _confusion_metrics(evaluation)
    member_scores = evaluation.member_scores.aggregate_scores
    nonmember_scores = evaluation.nonmember_scores.aggregate_scores
    node_types = nx.get_node_attributes(graph, "type")

    lines = [
        "FedMIA TARS Real Reactome DAG Smoke Report",
        f"report_id: {report_id}",
        "source_test: tests/test_tars_real_dag_smoke.py",
        "data: data/datasets/ProstateCancer/pnet_x_100.npy + pnet_y_100.npy",
        "dag: data/datasets/Reactome ReactomePathwaysRelation + UniProt2Reactome",
        "measurement: Eq. 7 gradient cosine between client update and target sample gradient",
        "",
        "Graph summary",
        f"  graph_name: {graph.name}",
        f"  nodes: {graph.number_of_nodes()}",
        f"  edges: {graph.number_of_edges()}",
        f"  feature_input_nodes: {sum(1 for value in node_types.values() if value == 'Feature')}",
        f"  protein_adapter_nodes: {sum(1 for value in node_types.values() if value == 'Protein')}",
        f"  reactome_term_nodes: {sum(1 for value in node_types.values() if value == 'Term')}",
        f"  selected_pnet_columns: {len(selected_features)}",
        "",
        "Run configuration",
        f"  clients: {config['num_clients']}",
        f"  communication_rounds: {config['rounds']}",
        f"  local_epochs_per_round: {config['local_epochs']}",
        f"  samples_per_client: {config['samples_per_client']}",
        f"  candidate_members: {config['candidate_count']}",
        f"  candidate_nonmembers: {config['candidate_count']}",
        f"  learning_rate: {config['learning_rate']}",
        f"  threshold_delta: {config['threshold']}",
        f"  elapsed_seconds: {elapsed:.2f}",
        "",
        "Training summary",
        f"  initial_combined_accuracy: {initial_acc:.4f}",
        f"  final_combined_accuracy: {final_acc:.4f}",
        "",
        "Attack metrics",
        f"  auc: {evaluation.auc:.6f}",
        f"  log_auc: {evaluation.log_auc:.6f}",
        f"  tpr_at_fpr_0.1: {evaluation.tprs['0.1']:.6f}",
        f"  tpr_at_fpr_0.01: {evaluation.tprs['0.01']:.6f}",
        f"  threshold_accuracy: {metrics['accuracy']:.6f}",
        f"  precision_at_delta: {metrics['precision']:.6f}",
        f"  recall_at_delta: {metrics['recall']:.6f}",
        f"  f1_at_delta: {metrics['f1']:.6f}",
        f"  true_positives: {metrics['tp']}",
        f"  false_positives: {metrics['fp']}",
        f"  true_negatives: {metrics['tn']}",
        f"  false_negatives: {metrics['fn']}",
        "",
        "Aggregate score distributions",
        f"  member_mean: {_mean(member_scores):.6f}",
        f"  nonmember_mean: {_mean(nonmember_scores):.6f}",
        "",
        "Communication/training logs",
    ]
    lines.extend(f"  {line}" for line in log_lines)
    lines.extend(
        [
            "",
            "Commentary",
            "  This smoke uses the actual Reactome DAG and aligned pnet feature columns. "
            "It is intentionally small, so metrics are plumbing checks rather than final claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_real_dag_smoke():
    start = time.time()
    _set_seed(20260509)
    device = torch.device("cpu")
    config = {
        "num_clients": 4,
        "rounds": 2,
        "local_epochs": 1,
        "samples_per_client": 8,
        "candidate_count": 4,
        "learning_rate": 0.03,
        "threshold": 0.5,
    }

    feature_index = _load_pnet_feature_index()
    graph, selected_features = build_real_reactome_feature_graph(feature_index)
    x, y = _load_real_pnet_chunk_for_feature_rows(
        selected_features["column_index"].to_numpy(),
        max_samples=64,
    )
    client_indices, holdout_indices = _make_client_splits(
        y,
        num_clients=config["num_clients"],
        samples_per_client=config["samples_per_client"],
        holdout_samples=config["candidate_count"],
        seed=20260509,
    )

    client_datasets = [TensorDataset(x[indices], y[indices]) for indices in client_indices]
    client_loaders = [
        DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(1200 + idx))
        for idx, dataset in enumerate(client_datasets)
    ]
    combined_dataset = TensorDataset(
        torch.cat([dataset.tensors[0] for dataset in client_datasets], dim=0),
        torch.cat([dataset.tensors[1] for dataset in client_datasets], dim=0),
    )
    combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False)
    member_samples = [client_datasets[0][idx] for idx in range(config["candidate_count"])]
    nonmember_samples = [(x[idx], y[idx]) for idx in holdout_indices]

    global_model = binn(graph=graph, output_size=1, dropout_prob=0.0, output_last_layers=1).to(device)
    initial_acc = _accuracy(global_model, combined_loader, device)

    member_rounds = []
    nonmember_rounds = []
    log_lines = []

    for round_id in range(1, config["rounds"] + 1):
        round_start = time.time()
        global_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        local_updates = []
        local_losses = []

        log_lines.append(
            f"round={round_id:02d} communication_start "
            f"global_acc={_accuracy(global_model, combined_loader, device):.4f}"
        )
        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_state, local_loss = _train_one_client(
                local_model,
                loader,
                device,
                local_epochs=config["local_epochs"],
                lr=config["learning_rate"],
                round_id=round_id,
                client_id=client_id,
                log_lines=log_lines,
            )
            local_states.append(local_state)
            local_losses.append(local_loss)
            parameter_names = [name for name, _ in local_model.named_parameters()]
            local_updates.append(_state_update(global_state, local_state, parameter_names))

        member_rounds.append(
            _measure_round(
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
            _measure_round(
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
        log_lines.append(
            f"round={round_id:02d} communication_end mean_client_loss={_mean(local_losses):.6f} "
            f"global_acc={_accuracy(global_model, combined_loader, device):.4f} "
            f"elapsed_seconds={time.time() - round_start:.2f}"
        )

    final_acc = _accuracy(global_model, combined_loader, device)
    evaluation = evaluate_membership(
        member_rounds,
        nonmember_rounds,
        config=FedMIAConfig(threshold=config["threshold"]),
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    report_id = f"test_tars_real_dag_smoke_{timestamp}_{uuid.uuid4().hex[:8]}"
    report_text = _build_report(
        report_id,
        graph,
        selected_features,
        config,
        log_lines,
        evaluation,
        initial_acc,
        final_acc,
        elapsed=time.time() - start,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{report_id}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    print(report_text)
    print(f"report_path: {report_path}")
    return evaluation, report_path, graph


class TARSRealDAGFedMIASmokeTest(unittest.TestCase):
    def test_real_reactome_dag_fedmia_smoke(self):
        evaluation, report_path, graph = run_real_dag_smoke()
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(nx.is_directed_acyclic_graph(graph))
        self.assertGreater(graph.number_of_nodes(), 0)
        self.assertGreaterEqual(evaluation.auc, 0.0)
        self.assertLessEqual(evaluation.auc, 1.0)


if __name__ == "__main__":
    run_real_dag_smoke()
