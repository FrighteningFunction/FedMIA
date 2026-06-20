import copy
import math
import os
import sys
import time
import uuid
import unittest
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import (
    FedMIAConfig,
    FedMIARoundMeasurements,
    cosine_measurement,
    evaluate_membership,
)
from models.binn import binn
from utils.federated import fed_avg_state_dicts


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data", "datasets", "ProstateCancer")
REPORT_DIR = os.path.join(REPO_ROOT, "reports")


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_real_pnet_chunk(max_samples=80):
    x_path = os.path.join(DATA_DIR, "pnet_x_100.npy")
    y_path = os.path.join(DATA_DIR, "pnet_y_100.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "Expected real BINN smoke chunk at "
            f"{os.path.relpath(x_path, REPO_ROOT)} and {os.path.relpath(y_path, REPO_ROOT)}"
        )

    x = np.load(x_path)[:max_samples].astype("float32", copy=False)
    y = np.load(y_path)[:max_samples].astype("float32", copy=False).reshape(-1, 1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)

    return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def _balanced_indices(labels, count, rng, exclude=None):
    exclude = set(exclude or [])
    label_values = labels.view(-1).cpu().numpy()
    selected = []

    for label in (0.0, 1.0):
        candidates = [idx for idx, value in enumerate(label_values) if value == label and idx not in exclude]
        rng.shuffle(candidates)
        selected.extend(candidates[: count // 2])

    if len(selected) < count:
        remaining = [idx for idx in range(len(label_values)) if idx not in exclude and idx not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])

    rng.shuffle(selected)
    return selected[:count]


def _make_client_splits(labels, num_clients, samples_per_client, holdout_samples, seed):
    rng = np.random.default_rng(seed)
    used = set()
    client_indices = []

    for _ in range(num_clients):
        indices = _balanced_indices(labels, samples_per_client, rng, used)
        used.update(indices)
        client_indices.append(indices)

    holdout_indices = _balanced_indices(labels, holdout_samples, rng, used)
    return client_indices, holdout_indices


def _binary_loss(logits, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.view_as(logits))


def _accuracy(model, loader, device):
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


def _train_one_client(model, loader, device, local_epochs, lr, round_id, client_id, log_lines):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    epoch_losses = []

    for epoch in range(local_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = _binary_loss(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.numel()
            pred = (torch.sigmoid(logits).view(-1) >= 0.5).float()
            truth = y.view(-1).float()
            running_correct += pred.eq(truth).sum().item()
            running_total += truth.numel()

        avg_loss = running_loss / max(running_total, 1)
        avg_acc = running_correct / max(running_total, 1)
        epoch_losses.append(avg_loss)
        log_lines.append(
            f"round={round_id:02d} client={client_id:02d} local_epoch={epoch + 1:02d} "
            f"loss={avg_loss:.6f} acc={avg_acc:.4f}"
        )

    return copy.deepcopy(model.state_dict()), float(sum(epoch_losses) / len(epoch_losses))


def _state_update(global_state, local_state, parameter_names):
    return {
        key: (global_state[key].detach().cpu() - local_state[key].detach().cpu())
        for key in parameter_names
    }


def _sample_gradient(global_model, sample, device):
    x, y = sample
    global_model.zero_grad(set_to_none=True)
    global_model.train(False)
    logits = global_model(x.unsqueeze(0).to(device))
    loss = _binary_loss(logits, y.view(1, 1).to(device))
    params = [(name, param) for name, param in global_model.named_parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, [param for _, param in params], allow_unused=True)
    return {
        name: (torch.zeros_like(param).detach().cpu() if grad is None else grad.detach().cpu())
        for (name, param), grad in zip(params, grads)
    }


def _measure_round(global_model, target_update, reference_updates, samples, device, round_id, split):
    gradients = [_sample_gradient(global_model, sample, device) for sample in samples]
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


def _mean(values):
    return sum(values) / max(len(values), 1)


def _safe_min(values):
    return min(values) if values else float("nan")


def _safe_max(values):
    return max(values) if values else float("nan")


def _build_report(report_id, config, log_lines, evaluation, initial_acc, final_acc, elapsed_seconds):
    member_scores = evaluation.member_scores.aggregate_scores
    nonmember_scores = evaluation.nonmember_scores.aggregate_scores
    member_predictions = evaluation.member_scores.predictions
    nonmember_predictions = evaluation.nonmember_scores.predictions
    true_positives = sum(member_predictions)
    false_negatives = len(member_predictions) - true_positives
    false_positives = sum(nonmember_predictions)
    true_negatives = len(nonmember_predictions) - false_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2.0 * precision * recall / max(precision + recall, 1e-12)
    threshold_acc = (
        true_positives + true_negatives
    ) / max(len(member_predictions) + len(nonmember_predictions), 1)

    if evaluation.auc >= 0.8:
        comment = "The smoke run shows a clear member/non-member separation on this real-data chunk."
    elif evaluation.auc >= 0.65:
        comment = (
            "The smoke run shows a useful but noisy separation, which is plausible for a tiny "
            "CPU chunk and few communication rounds."
        )
    else:
        comment = (
            "The smoke run is weak on this tiny chunk; the plumbing works, but more rounds, "
            "larger client shards, or a tuned threshold should be tested before drawing claims."
        )

    lines = [
        "FedMIA TARS BINN Smoke Report",
        f"report_id: {report_id}",
        f"source_test: tests/test_tars_smoke.py",
        f"data: data/datasets/ProstateCancer/pnet_x_100.npy + pnet_y_100.npy",
        f"measurement: Eq. 7 gradient cosine between client update and target sample gradient",
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
        f"  elapsed_seconds: {elapsed_seconds:.2f}",
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
        f"  tpr_at_fpr_0.001: {evaluation.tprs['0.001']:.6f}",
        f"  threshold_accuracy: {threshold_acc:.6f}",
        f"  precision_at_delta: {precision:.6f}",
        f"  recall_at_delta: {recall:.6f}",
        f"  f1_at_delta: {f1_score:.6f}",
        f"  true_positives: {true_positives}",
        f"  false_positives: {false_positives}",
        f"  true_negatives: {true_negatives}",
        f"  false_negatives: {false_negatives}",
        f"  predicted_members_for_true_members: {true_positives}/{len(member_predictions)}",
        f"  predicted_nonmembers_for_true_nonmembers: {true_negatives}/{len(nonmember_predictions)}",
        "",
        "Aggregate score distributions",
        f"  member_mean: {_mean(member_scores):.6f}",
        f"  member_min: {_safe_min(member_scores):.6f}",
        f"  member_max: {_safe_max(member_scores):.6f}",
        f"  nonmember_mean: {_mean(nonmember_scores):.6f}",
        f"  nonmember_min: {_safe_min(nonmember_scores):.6f}",
        f"  nonmember_max: {_safe_max(nonmember_scores):.6f}",
        "",
        "Per-round score means",
    ]

    for round_idx, (member_round, nonmember_round) in enumerate(
        zip(evaluation.member_scores.per_round_scores, evaluation.nonmember_scores.per_round_scores),
        start=1,
    ):
        lines.append(
            f"  round={round_idx:02d} member_mean={_mean(member_round):.6f} "
            f"nonmember_mean={_mean(nonmember_round):.6f}"
        )

    lines.extend(["", "Communication/training logs"])
    lines.extend(f"  {line}" for line in log_lines)
    lines.extend(["", "Commentary", f"  {comment}"])
    return "\n".join(lines) + "\n"


def run_tars_smoke():
    start = time.time()
    _set_seed(20260507)
    device = torch.device("cpu")

    config = {
        "num_clients": 4,
        "rounds": 3,
        "local_epochs": 2,
        "samples_per_client": 12,
        "candidate_count": 6,
        "learning_rate": 0.04,
        "threshold": 0.5,
    }
    log_lines = []

    x, y = _load_real_pnet_chunk(max_samples=80)
    client_indices, holdout_indices = _make_client_splits(
        y,
        num_clients=config["num_clients"],
        samples_per_client=config["samples_per_client"],
        holdout_samples=config["candidate_count"],
        seed=20260507,
    )
    client_datasets = [TensorDataset(x[indices], y[indices]) for indices in client_indices]
    client_loaders = [
        DataLoader(dataset, batch_size=6, shuffle=True, generator=torch.Generator().manual_seed(900 + idx))
        for idx, dataset in enumerate(client_datasets)
    ]
    combined_dataset = TensorDataset(
        torch.cat([dataset.tensors[0] for dataset in client_datasets], dim=0),
        torch.cat([dataset.tensors[1] for dataset in client_datasets], dim=0),
    )
    combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False)

    member_samples = [client_datasets[0][idx] for idx in range(config["candidate_count"])]
    nonmember_samples = [(x[idx], y[idx]) for idx in holdout_indices]

    global_model = binn(
        input_size=x.shape[1],
        output_size=1,
        hidden_layers="16,8",
        dropout_prob=0.0,
        output_last_layers=1,
    ).to(device)
    initial_acc = _accuracy(global_model, combined_loader, device)

    member_rounds = []
    nonmember_rounds = []

    for round_id in range(1, config["rounds"] + 1):
        round_start = time.time()
        global_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        local_updates = []
        local_losses = []

        log_lines.append(f"round={round_id:02d} communication_start global_acc={_accuracy(global_model, combined_loader, device):.4f}")
        for client_id, loader in enumerate(client_loaders):
            local_model = binn(
                input_size=x.shape[1],
                output_size=1,
                hidden_layers="16,8",
                dropout_prob=0.0,
                output_last_layers=1,
            ).to(device)
            local_model.load_state_dict(global_state)
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
    report_id = f"test_tars_smoke_{timestamp}_{uuid.uuid4().hex[:8]}"
    report_text = _build_report(
        report_id,
        config,
        log_lines,
        evaluation,
        initial_acc,
        final_acc,
        elapsed_seconds=time.time() - start,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"{report_id}.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    print(report_text)
    print(f"report_path: {report_path}")

    if not math.isfinite(evaluation.auc):
        raise AssertionError("FedMIA AUC must be finite.")
    if not math.isfinite(evaluation.log_auc):
        raise AssertionError("FedMIA log AUC must be finite.")
    if len(evaluation.member_scores.aggregate_scores) != config["candidate_count"]:
        raise AssertionError("Unexpected number of member scores.")
    if len(evaluation.nonmember_scores.aggregate_scores) != config["candidate_count"]:
        raise AssertionError("Unexpected number of nonmember scores.")

    return evaluation, report_path


class TARSFedMIASmokeTest(unittest.TestCase):
    def test_real_binn_chunk_fedmia_smoke(self):
        evaluation, report_path = run_tars_smoke()

        self.assertTrue(os.path.exists(report_path))
        self.assertGreaterEqual(evaluation.auc, 0.0)
        self.assertLessEqual(evaluation.auc, 1.0)
        self.assertIn("0.1", evaluation.tprs)
        self.assertIn("0.01", evaluation.tprs)


if __name__ == "__main__":
    run_tars_smoke()
