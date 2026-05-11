from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import random
import sys
import tarfile
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import FedMIAConfig, FedMIARoundMeasurements, evaluate_membership  # noqa: E402
from models.alexnet import alexnet  # noqa: E402
from utils.federated import fed_avg_state_dicts  # noqa: E402


LOGGER = logging.getLogger("fedmia_cifar100_alexnet_grid")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
REPORT_DIR = os.path.join(REPO_ROOT, "reports")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
DATA_DIR = os.path.join(REPO_ROOT, "data", "datasets", "CIFAR100")

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_FOLDER = "cifar-100-python"
CIFAR100_CLASSES = 100
CIFAR100_MEAN = torch.tensor([0.507, 0.487, 0.441]).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.267, 0.256, 0.276]).view(1, 3, 1, 1)


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
class RunResult:
    config: GridConfig
    run_id: int
    seed: int
    train_acc_initial: float
    train_acc_final: float
    test_acc_initial: float
    test_acc_final: float
    member_count: int
    nonmember_count: int
    client_sizes: List[int]
    elapsed_seconds: float
    metrics: Dict[str, float]
    member_scores: Dict[str, List[float]]
    nonmember_scores: Dict[str, List[float]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FedMIA-I/FedMIA-II reproduction grid for AlexNet on CIFAR100."
    )
    parser.add_argument("--runs", type=int, default=10, help="independent trajectories per grid cell")
    parser.add_argument("--client-grid", default="5,10", help="comma-separated FL client counts")
    parser.add_argument("--round-grid", default="100,200,300", help="comma-separated communication rounds")
    parser.add_argument("--local-epoch-grid", default="1,3,5", help="comma-separated local epochs")
    parser.add_argument(
        "--beta-grid",
        default="iid,10,1,0.1",
        help="comma-separated Dirichlet beta values; use iid for beta=infinity",
    )
    parser.add_argument(
        "--sample-fraction-grid",
        default="1.0",
        help="fractions of the max disjoint CIFAR100 samples/client; 1.0 gives 10000 for 5 clients and 5000 for 10",
    )
    parser.add_argument(
        "--samples-per-client-grid",
        default="auto",
        help="absolute samples/client grid, or auto to use sample fractions",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=512,
        help="member and nonmember candidates attacked per trajectory; increase for final runs if time allows",
    )
    parser.add_argument(
        "--attack-every",
        type=int,
        default=1,
        help="measure FedMIA every N communication rounds; 1 matches Algorithm 1",
    )
    parser.add_argument(
        "--nonmember-other-client-fraction",
        type=float,
        default=0.1,
        help="fraction of non-target-client training samples included in the nonmember pool",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="local training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="utility/loss evaluation batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="paper initial SGD learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.99, help="paper learning-rate decay per communication round")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="SGD weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--threshold", type=float, default=0.5, help="FedMIA delta threshold")
    parser.add_argument(
        "--threshold-grid",
        default="0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="comma-separated deltas for best-F1 diagnostics",
    )
    parser.add_argument("--outlier-std-factor", type=float, default=3.0, help="FedMIA 3-sigma filter")
    parser.add_argument("--min-variance", type=float, default=1e-8, help="FedMIA Gaussian variance floor")
    parser.add_argument("--seed", type=int, default=20260511, help="base random seed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="training device")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--data-dir", default=DATA_DIR, help="directory containing cifar-100-python")
    parser.add_argument("--download", action="store_true", help="download CIFAR100 if missing")
    parser.add_argument("--max-train-samples", type=int, default=0, help="debug cap; 0 uses all 50000")
    parser.add_argument("--max-test-samples", type=int, default=0, help="debug cap; 0 uses all 10000")
    parser.add_argument("--max-configs", type=int, default=0, help="debug cap on grid cells; 0 uses all")
    parser.add_argument("--synthetic-data", action="store_true", help="plumbing only: use random CIFAR-shaped tensors")
    parser.add_argument("--log-client-epochs", action="store_true", help="log every client local epoch at INFO")
    parser.add_argument("--report-dir", default=REPORT_DIR, help="directory for text/CSV reports")
    parser.add_argument("--log-dir", default=LOG_DIR, help="directory for execution logs")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def make_experiment_id() -> str:
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"fedmia_cifar100_alexnet_{stamp}_{uuid.uuid4().hex[:8]}"


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
    parsed = []
    for part in value.split(","):
        label = part.strip()
        if not label:
            continue
        if label.lower() in {"iid", "inf", "infinity"}:
            parsed.append(("iid", None))
        else:
            parsed.append((label, float(label)))
    return parsed


def md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_download_cifar100(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    folder = os.path.join(data_dir, CIFAR100_FOLDER)
    if os.path.exists(os.path.join(folder, "train")) and os.path.exists(os.path.join(folder, "test")):
        return
    archive_path = os.path.join(data_dir, CIFAR100_ARCHIVE)
    if not os.path.exists(archive_path):
        LOGGER.info("Downloading CIFAR100 to %s", archive_path)
        urllib.request.urlretrieve(CIFAR100_URL, archive_path)
    if md5(archive_path) != CIFAR100_MD5:
        raise ValueError(f"CIFAR100 archive checksum mismatch: {archive_path}")
    LOGGER.info("Extracting CIFAR100 archive to %s", data_dir)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)


def read_cifar100_split(data_dir: str, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    path = os.path.join(data_dir, CIFAR100_FOLDER, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing CIFAR100 file {path}. Place cifar-100-python under {data_dir} "
            "or rerun with DOWNLOAD=1."
        )
    with open(path, "rb") as handle:
        entry = pickle.load(handle, encoding="latin1")
    data = entry["data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    labels = np.asarray(entry["fine_labels"], dtype="int64")
    x = torch.from_numpy(data)
    x = (x - CIFAR100_MEAN) / CIFAR100_STD
    y = torch.from_numpy(labels)
    return x.contiguous(), y.contiguous()


def load_cifar100(args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.synthetic_data:
        rng = np.random.default_rng(args.seed)
        train_count = args.max_train_samples if args.max_train_samples > 0 else 500
        test_count = args.max_test_samples if args.max_test_samples > 0 else 100
        train_x = torch.from_numpy(rng.normal(size=(train_count, 3, 32, 32)).astype("float32"))
        train_y = torch.from_numpy(rng.integers(0, CIFAR100_CLASSES, size=train_count, dtype=np.int64))
        test_x = torch.from_numpy(rng.normal(size=(test_count, 3, 32, 32)).astype("float32"))
        test_y = torch.from_numpy(rng.integers(0, CIFAR100_CLASSES, size=test_count, dtype=np.int64))
        return train_x, train_y, test_x, test_y
    if args.download:
        maybe_download_cifar100(args.data_dir)
    train_x, train_y = read_cifar100_split(args.data_dir, "train")
    test_x, test_y = read_cifar100_split(args.data_dir, "test")
    if args.max_train_samples and args.max_train_samples > 0:
        train_x = train_x[: args.max_train_samples].contiguous()
        train_y = train_y[: args.max_train_samples].contiguous()
    if args.max_test_samples and args.max_test_samples > 0:
        test_x = test_x[: args.max_test_samples].contiguous()
        test_y = test_y[: args.max_test_samples].contiguous()
    return train_x, train_y, test_x, test_y


def balanced_sample_indices(labels: torch.Tensor, pool: Iterable[int], count: int, rng) -> List[int]:
    pool = list(pool)
    if count <= 0:
        return []
    if len(pool) < count:
        raise ValueError(f"Not enough samples in pool: requested {count}, available {len(pool)}")
    label_values = labels.detach().cpu().numpy()
    classes = sorted(np.unique(label_values[pool]).tolist())
    selected: List[int] = []
    per_class = count // max(len(classes), 1)
    for label in classes:
        candidates = [idx for idx in pool if int(label_values[idx]) == int(label)]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    if len(selected) < count:
        remaining = [idx for idx in pool if idx not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected[:count]


def make_iid_clients(labels, clients: int, samples_per_client: int, rng) -> List[List[int]]:
    total_count = clients * samples_per_client
    all_indices = list(range(len(labels)))
    selected = balanced_sample_indices(labels, all_indices, total_count, rng)
    label_values = labels.detach().cpu().numpy()
    by_class = {label: [] for label in sorted(np.unique(label_values[selected]).tolist())}
    for idx in selected:
        by_class[int(label_values[idx])].append(idx)
    for indices in by_class.values():
        rng.shuffle(indices)
    client_indices = [[] for _ in range(clients)]
    for indices in by_class.values():
        offset = int(rng.integers(0, clients))
        for pos, idx in enumerate(indices):
            client_indices[(offset + pos) % clients].append(idx)
    for indices in client_indices:
        rng.shuffle(indices)
    for client_id, indices in enumerate(client_indices):
        if indices:
            continue
        donor_id = max(range(clients), key=lambda idx: len(client_indices[idx]))
        if len(client_indices[donor_id]) <= 1:
            raise ValueError("IID split produced empty clients with no donor available.")
        move_at = int(rng.integers(0, len(client_indices[donor_id])))
        indices.append(client_indices[donor_id].pop(move_at))
        LOGGER.warning("IID split produced empty client=%s; moved one sample from client=%s.", client_id, donor_id)
    return client_indices


def make_dirichlet_clients(labels, clients: int, samples_per_client: int, beta: float, rng) -> List[List[int]]:
    total_count = clients * samples_per_client
    all_indices = list(range(len(labels)))
    selected = balanced_sample_indices(labels, all_indices, total_count, rng)
    label_values = labels.detach().cpu().numpy()
    client_indices = [[] for _ in range(clients)]
    for label in sorted(np.unique(label_values[selected]).tolist()):
        class_indices = [idx for idx in selected if int(label_values[idx]) == int(label)]
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
    for client_id, indices in enumerate(client_indices):
        rng.shuffle(indices)
        if indices:
            continue
        donor_id = max(range(clients), key=lambda idx: len(client_indices[idx]))
        if len(client_indices[donor_id]) <= 1:
            raise ValueError("Dirichlet split produced empty clients with no donor available.")
        move_at = int(rng.integers(0, len(client_indices[donor_id])))
        indices.append(client_indices[donor_id].pop(move_at))
        LOGGER.warning("Dirichlet split produced empty client=%s; moved one sample from client=%s.", client_id, donor_id)
    return client_indices


def make_client_indices(labels, config: GridConfig, rng) -> List[List[int]]:
    if config.beta is None:
        return make_iid_clients(labels, config.clients, config.samples_per_client, rng)
    return make_dirichlet_clients(labels, config.clients, config.samples_per_client, config.beta, rng)


def sample_mixed_nonmember_refs(
    train_y,
    test_y,
    non_target_train_indices: Sequence[int],
    count: int,
    other_client_fraction: float,
    rng,
) -> List[Tuple[str, int]]:
    sampled_non_target_count = int(round(len(non_target_train_indices) * other_client_fraction))
    sampled_non_target_count = min(max(sampled_non_target_count, 0), len(non_target_train_indices))
    train_part = []
    if sampled_non_target_count > 0:
        shuffled = list(non_target_train_indices)
        rng.shuffle(shuffled)
        train_part = [("train", idx) for idx in shuffled[:sampled_non_target_count]]
    pool = [("test", idx) for idx in range(len(test_y))] + train_part
    if len(pool) < count:
        raise ValueError(f"Nonmember pool too small: requested {count}, available {len(pool)}")
    label_by_pos = []
    for split, idx in pool:
        label_by_pos.append(int(test_y[idx].item()) if split == "test" else int(train_y[idx].item()))
    selected_positions: List[int] = []
    classes = sorted(set(label_by_pos))
    per_class = count // max(len(classes), 1)
    for label in classes:
        candidates = [pos for pos, value in enumerate(label_by_pos) if value == label]
        rng.shuffle(candidates)
        selected_positions.extend(candidates[:per_class])
    if len(selected_positions) < count:
        selected_set = set(selected_positions)
        remaining = [pos for pos in range(len(pool)) if pos not in selected_set]
        rng.shuffle(remaining)
        selected_positions.extend(remaining[: count - len(selected_positions)])
    rng.shuffle(selected_positions)
    return [pool[pos] for pos in selected_positions[:count]]


def collect_train_samples(train_x, train_y, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return train_x[index_tensor].contiguous(), train_y[index_tensor].contiguous()


def collect_mixed_samples(train_x, train_y, test_x, test_y, refs: Sequence[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for split, idx in refs:
        if split == "test":
            xs.append(test_x[idx])
            ys.append(test_y[idx])
        else:
            xs.append(train_x[idx])
            ys.append(train_y[idx])
    return torch.stack(xs).contiguous(), torch.stack(ys).contiguous()


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
        if explicit_counts is None:
            sample_counts = sorted(
                {
                    max(1, min(max_per_client, int(round(max_per_client * fraction))))
                    for fraction in sample_fractions
                }
            )
            fraction_by_count = {count: min(1.0, count / max(max_per_client, 1)) for count in sample_counts}
        else:
            sample_counts = []
            fraction_by_count = {}
            for count in explicit_counts:
                if count > max_per_client:
                    LOGGER.warning(
                        "Skipping samples_per_client=%s for clients=%s; max disjoint count is %s.",
                        count,
                        clients,
                        max_per_client,
                    )
                    continue
                sample_counts.append(count)
                fraction_by_count[count] = None
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
                                sample_fraction=fraction_by_count[samples_per_client],
                                samples_per_client=samples_per_client,
                            )
                        )
                        config_id += 1
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]
    return configs


def make_loaders(train_x, train_y, test_x, test_y, client_indices, args, seed):
    generator = torch.Generator().manual_seed(seed)
    train_base = TensorDataset(train_x, train_y)
    client_datasets = [Subset(train_base, indices) for indices in client_indices]
    client_loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
        )
        for dataset in client_datasets
    ]
    train_eval_loader = DataLoader(
        ConcatDataset(client_datasets),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    return client_datasets, client_loaders, train_eval_loader, test_loader


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train_one_client(model, loader, device, args, lr: float, run_id: int, config_id: int, round_id: int, client_id: int, jsonl_path: str):
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    epoch_losses = []
    epoch_accs = []
    for local_epoch in range(1, args.local_epochs_current + 1):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_correct += logits.argmax(dim=1).eq(y).sum().item()
            total += y.numel()
        avg_loss = total_loss / max(total, 1)
        avg_acc = total_correct / max(total, 1)
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
        if args.log_client_epochs:
            LOGGER.info(
                "config=%03d run=%03d round=%03d client=%02d local_epoch=%02d loss=%.6f acc=%.4f",
                config_id,
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
                "config_id": config_id,
                "run": run_id,
                "round": round_id,
                "client": client_id,
                "local_epoch": local_epoch,
                "loss": avg_loss,
                "accuracy": avg_acc,
                "lr": lr,
            },
        )
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in model.state_dict().items()
    }, float(np.mean(epoch_losses)), float(np.mean(epoch_accs))


def vector_from_state_delta(global_state, local_state, parameter_names, device) -> torch.Tensor:
    parts = []
    for name in parameter_names:
        delta = global_state[name] - local_state[name]
        parts.append(delta.reshape(-1).to(device, non_blocking=True))
    return torch.cat(parts)


def gradient_vector(model, x: torch.Tensor, y: torch.Tensor, device) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    model.eval()
    logits = model(x.unsqueeze(0).to(device))
    loss = torch.nn.functional.cross_entropy(logits, y.view(1).to(device))
    params = [param for param in model.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    vector_parts = [
        torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1)
        for grad, param in zip(grads, params)
    ]
    return torch.cat(vector_parts).detach()


def cosine_round_measurements(
    global_model,
    update_matrix: torch.Tensor,
    update_norms: torch.Tensor,
    samples_x: torch.Tensor,
    samples_y: torch.Tensor,
    device,
) -> Tuple[List[float], List[List[float]]]:
    target_values: List[float] = []
    reference_values: List[List[float]] = [[] for _ in range(update_matrix.shape[0] - 1)]
    for idx in range(samples_y.numel()):
        grad_vec = gradient_vector(global_model, samples_x[idx], samples_y[idx], device)
        grad_norm = torch.linalg.vector_norm(grad_vec).clamp_min(1e-12)
        scores = torch.mv(update_matrix, grad_vec) / (update_norms * grad_norm)
        scores = scores.detach().cpu().tolist()
        target_values.append(float(scores[0]))
        for ref_idx, value in enumerate(scores[1:]):
            reference_values[ref_idx].append(float(value))
    return target_values, reference_values


def negative_loss_measurements(model, samples_x, samples_y, device, batch_size: int) -> List[float]:
    model.eval()
    values: List[float] = []
    loader = DataLoader(
        TensorDataset(samples_x, samples_y),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            values.extend((-losses).detach().cpu().tolist())
    return [float(value) for value in values]


def confusion_metrics(member_scores, nonmember_scores, threshold: float) -> Dict[str, float]:
    member_predictions = [1 if score > threshold else 0 for score in member_scores]
    nonmember_predictions = [1 if score > threshold else 0 for score in nonmember_scores]
    tp = sum(member_predictions)
    fn = len(member_predictions) - tp
    fp = sum(nonmember_predictions)
    tn = len(nonmember_predictions) - fp
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)
    return {
        "threshold": threshold,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fn / max(tp + fn, 1),
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "threshold_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
    }


def roc_curve(member_scores, nonmember_scores):
    labeled = [(float(score), 1) for score in member_scores] + [(float(score), 0) for score in nonmember_scores]
    labeled.sort(key=lambda item: item[0], reverse=True)
    positives = max(len(member_scores), 1)
    negatives = max(len(nonmember_scores), 1)
    tps = 0
    fps = 0
    fprs = [0.0]
    tprs = [0.0]
    for _, label in labeled:
        if label == 1:
            tps += 1
        else:
            fps += 1
        fprs.append(fps / negatives)
        tprs.append(tps / positives)
    if fprs[-1] != 1.0 or tprs[-1] != 1.0:
        fprs.append(1.0)
        tprs.append(1.0)
    return fprs, tprs


def trapezoid_auc(xs, ys) -> float:
    return float(sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0 for i in range(1, len(xs))))


def tpr_at_fpr(fprs, tprs, threshold: float) -> float:
    best = 0.0
    for fpr, tpr in zip(fprs, tprs):
        if fpr < threshold:
            best = tpr
    return best


def parse_threshold_grid(value: str) -> List[float]:
    return sorted({float(part.strip()) for part in value.split(",") if part.strip()})


def score_metrics(member_scores, nonmember_scores, threshold: float, threshold_grid: Sequence[float]) -> Dict[str, float]:
    metrics = confusion_metrics(member_scores, nonmember_scores, threshold)
    fprs, tprs = roc_curve(member_scores, nonmember_scores)
    sweep_rows = [confusion_metrics(member_scores, nonmember_scores, item) for item in threshold_grid]
    sweep_rows.sort(key=lambda row: (row["f1"], row["tnr"], row["tpr"]), reverse=True)
    best = sweep_rows[0] if sweep_rows else metrics
    metrics.update(
        {
            "auc": trapezoid_auc(fprs, tprs),
            "tpr_at_fpr_0.1": tpr_at_fpr(fprs, tprs, 0.1),
            "tpr_at_fpr_0.01": tpr_at_fpr(fprs, tprs, 0.01),
            "tpr_at_fpr_0.001": tpr_at_fpr(fprs, tprs, 0.001),
            "best_f1": best["f1"],
            "best_f1_threshold": best["threshold"],
            "best_f1_tpr": best["tpr"],
            "best_f1_tnr": best["tnr"],
            "member_score_mean": float(np.mean(member_scores)) if member_scores else float("nan"),
            "nonmember_score_mean": float(np.mean(nonmember_scores)) if nonmember_scores else float("nan"),
        }
    )
    return metrics


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_rounds(member_rounds, nonmember_rounds, config, threshold_grid):
    evaluation = evaluate_membership(member_rounds, nonmember_rounds, config)
    metrics = score_metrics(
        evaluation.member_scores.aggregate_scores,
        evaluation.nonmember_scores.aggregate_scores,
        config.threshold,
        threshold_grid,
    )
    metrics["auc"] = evaluation.auc
    metrics["tpr_at_fpr_0.001"] = evaluation.tprs.get("0.001", metrics["tpr_at_fpr_0.001"])
    metrics["tpr_at_fpr_0.0001"] = evaluation.tprs.get("0.0001", 0.0)
    return evaluation, metrics


def run_one_trajectory(config: GridConfig, run_id: int, run_seed: int, train_x, train_y, test_x, test_y, args, device, jsonl_path: str) -> RunResult:
    run_start = time.time()
    set_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    client_indices = make_client_indices(train_y, config, rng)
    member_count = min(args.candidate_count, len(client_indices[0]))
    member_indices = balanced_sample_indices(train_y, client_indices[0], member_count, rng)
    non_target_indices = [idx for indices in client_indices[1:] for idx in indices]
    nonmember_refs = sample_mixed_nonmember_refs(
        train_y,
        test_y,
        non_target_indices,
        member_count,
        args.nonmember_other_client_fraction,
        rng,
    )
    member_x, member_y = collect_train_samples(train_x, train_y, member_indices)
    nonmember_x, nonmember_y = collect_mixed_samples(train_x, train_y, test_x, test_y, nonmember_refs)
    client_datasets, client_loaders, train_eval_loader, test_loader = make_loaders(
        train_x, train_y, test_x, test_y, client_indices, args, run_seed
    )
    global_model = alexnet(num_classes=CIFAR100_CLASSES).to(device)
    parameter_names = [name for name, _ in global_model.named_parameters()]
    train_acc_initial = accuracy(global_model, train_eval_loader, device)
    test_acc_initial = accuracy(global_model, test_loader, device)
    loss_member_rounds: List[FedMIARoundMeasurements] = []
    loss_nonmember_rounds: List[FedMIARoundMeasurements] = []
    cosine_member_rounds: List[FedMIARoundMeasurements] = []
    cosine_nonmember_rounds: List[FedMIARoundMeasurements] = []
    run_args = argparse.Namespace(**vars(args))
    run_args.local_epochs_current = config.local_epochs

    LOGGER.info(
        "config=%03d run=%03d start seed=%s clients=%s rounds=%s local_epochs=%s beta=%s "
        "samples_per_client=%s candidates=%s train_acc=%.4f test_acc=%.4f client_sizes=%s",
        config.config_id,
        run_id,
        run_seed,
        config.clients,
        config.rounds,
        config.local_epochs,
        config.beta_label,
        config.samples_per_client,
        member_count,
        train_acc_initial,
        test_acc_initial,
        [len(indices) for indices in client_indices],
    )

    for round_id in range(1, config.rounds + 1):
        round_start = time.time()
        lr = args.lr * (args.lr_decay ** (round_id - 1))
        global_state = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in global_model.state_dict().items()
        }
        local_states = []
        local_update_vectors = []
        local_losses = []
        local_accs = []
        loss_member_target = None
        loss_nonmember_target = None
        loss_member_refs = []
        loss_nonmember_refs = []

        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            local_state, local_loss, local_acc = train_one_client(
                local_model,
                loader,
                device,
                run_args,
                lr,
                run_id,
                config.config_id,
                round_id,
                client_id,
                jsonl_path,
            )
            local_states.append(local_state)
            local_losses.append(local_loss)
            local_accs.append(local_acc)
            local_update_vectors.append(
                vector_from_state_delta(global_state, local_state, parameter_names, device)
            )
            if round_id % args.attack_every == 0:
                member_losses = negative_loss_measurements(
                    local_model, member_x, member_y, device, args.eval_batch_size
                )
                nonmember_losses = negative_loss_measurements(
                    local_model, nonmember_x, nonmember_y, device, args.eval_batch_size
                )
                if client_id == 0:
                    loss_member_target = member_losses
                    loss_nonmember_target = nonmember_losses
                else:
                    loss_member_refs.append(member_losses)
                    loss_nonmember_refs.append(nonmember_losses)
            del local_model

        if round_id % args.attack_every == 0:
            update_matrix = torch.stack(local_update_vectors).to(device)
            update_norms = torch.linalg.vector_norm(update_matrix, dim=1).clamp_min(1e-12)
            member_target, member_refs = cosine_round_measurements(
                global_model,
                update_matrix,
                update_norms,
                member_x,
                member_y,
                device,
            )
            nonmember_target, nonmember_refs_cos = cosine_round_measurements(
                global_model,
                update_matrix,
                update_norms,
                nonmember_x,
                nonmember_y,
                device,
            )
            cosine_member_rounds.append(
                FedMIARoundMeasurements(member_target, member_refs, round_id=round_id)
            )
            cosine_nonmember_rounds.append(
                FedMIARoundMeasurements(nonmember_target, nonmember_refs_cos, round_id=round_id)
            )
            loss_member_rounds.append(
                FedMIARoundMeasurements(loss_member_target or [], loss_member_refs, round_id=round_id)
            )
            loss_nonmember_rounds.append(
                FedMIARoundMeasurements(loss_nonmember_target or [], loss_nonmember_refs, round_id=round_id)
            )

        averaged_state = fed_avg_state_dicts(local_states, [len(dataset) for dataset in client_datasets])
        global_model.load_state_dict(averaged_state)
        round_elapsed = time.time() - round_start
        if round_id == 1 or round_id == config.rounds or round_id % max(1, config.rounds // 10) == 0:
            train_acc = accuracy(global_model, train_eval_loader, device)
            test_acc = accuracy(global_model, test_loader, device)
            LOGGER.info(
                "config=%03d run=%03d round=%03d/%03d lr=%.6f train_acc=%.4f test_acc=%.4f "
                "mean_client_loss=%.6f mean_client_acc=%.4f attack_rounds=%s elapsed=%.2fs",
                config.config_id,
                run_id,
                round_id,
                config.rounds,
                lr,
                train_acc,
                test_acc,
                float(np.mean(local_losses)),
                float(np.mean(local_accs)),
                len(loss_member_rounds),
                round_elapsed,
            )
        append_jsonl(
            jsonl_path,
            {
                "event": "communication_round",
                "config_id": config.config_id,
                "run": run_id,
                "round": round_id,
                "lr": lr,
                "mean_client_loss": float(np.mean(local_losses)),
                "mean_client_acc": float(np.mean(local_accs)),
                "attack_measured": round_id % args.attack_every == 0,
                "elapsed_seconds": round_elapsed,
            },
        )
        del local_update_vectors

    train_acc_final = accuracy(global_model, train_eval_loader, device)
    test_acc_final = accuracy(global_model, test_loader, device)
    fedmia_config = FedMIAConfig(
        threshold=args.threshold,
        outlier_std_factor=args.outlier_std_factor,
        min_variance=args.min_variance,
    )
    threshold_grid = parse_threshold_grid(args.threshold_grid)
    loss_eval, loss_metrics = evaluate_rounds(loss_member_rounds, loss_nonmember_rounds, fedmia_config, threshold_grid)
    cosine_eval, cosine_metrics = evaluate_rounds(cosine_member_rounds, cosine_nonmember_rounds, fedmia_config, threshold_grid)
    metrics = {}
    metrics.update(prefix_metrics("fedmia_i_loss", loss_metrics))
    metrics.update(prefix_metrics("fedmia_ii_cosine", cosine_metrics))
    elapsed = time.time() - run_start
    LOGGER.info(
        "config=%03d run=%03d finished train_acc=%.4f test_acc=%.4f "
        "fedmia_i_auc=%.4f fedmia_i_tpr_at_fpr_0.001=%.4f fedmia_i_f1=%.4f "
        "fedmia_ii_auc=%.4f fedmia_ii_tpr_at_fpr_0.001=%.4f fedmia_ii_f1=%.4f elapsed=%.2fs",
        config.config_id,
        run_id,
        train_acc_final,
        test_acc_final,
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
            "test_acc_initial": test_acc_initial,
            "test_acc_final": test_acc_final,
            "member_count": member_count,
            "nonmember_count": member_count,
            "client_sizes": [len(indices) for indices in client_indices],
            "elapsed_seconds": elapsed,
            **metrics,
        },
    )
    return RunResult(
        config=config,
        run_id=run_id,
        seed=run_seed,
        train_acc_initial=train_acc_initial,
        train_acc_final=train_acc_final,
        test_acc_initial=test_acc_initial,
        test_acc_final=test_acc_final,
        member_count=member_count,
        nonmember_count=member_count,
        client_sizes=[len(indices) for indices in client_indices],
        elapsed_seconds=elapsed,
        metrics=metrics,
        member_scores={
            "fedmia_i_loss": loss_eval.member_scores.aggregate_scores,
            "fedmia_ii_cosine": cosine_eval.member_scores.aggregate_scores,
        },
        nonmember_scores={
            "fedmia_i_loss": loss_eval.nonmember_scores.aggregate_scores,
            "fedmia_ii_cosine": cosine_eval.nonmember_scores.aggregate_scores,
        },
    )


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1 if len(arr) > 1 else 0))


def pooled_metrics(results: Sequence[RunResult], measurement: str, args) -> Dict[str, float]:
    member_scores: List[float] = []
    nonmember_scores: List[float] = []
    for result in results:
        member_scores.extend(result.member_scores[measurement])
        nonmember_scores.extend(result.nonmember_scores[measurement])
    metrics = score_metrics(
        member_scores,
        nonmember_scores,
        args.threshold,
        parse_threshold_grid(args.threshold_grid),
    )
    metrics["member_count"] = float(len(member_scores))
    metrics["nonmember_count"] = float(len(nonmember_scores))
    return metrics


def summarize_config(results: Sequence[RunResult], args) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for key in [
        "train_acc_initial",
        "train_acc_final",
        "test_acc_initial",
        "test_acc_final",
        "elapsed_seconds",
        "member_count",
        "nonmember_count",
    ]:
        mean, std = mean_std([float(getattr(result, key)) for result in results])
        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std
    for key in sorted(results[0].metrics.keys()):
        mean, std = mean_std([result.metrics[key] for result in results])
        row[f"{key}_mean"] = mean
        row[f"{key}_std"] = std
    for measurement in ("fedmia_i_loss", "fedmia_ii_cosine"):
        pooled = pooled_metrics(results, measurement, args)
        for key, value in pooled.items():
            row[f"{measurement}_pooled_{key}"] = value
    return row


def write_csv(path: str, rows: Sequence[Dict]):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, experiment_id, log_path, jsonl_path, csv_path, args, data_summary, rows, elapsed):
    lines = [
        "FedMIA CIFAR100 AlexNet Grid Report",
        f"experiment_id: {experiment_id}",
        "script: experiments/fedmia_cifar100_alexnet_grid.py",
        f"log_file: {os.path.relpath(log_path, REPO_ROOT)}",
        f"jsonl_log_file: {os.path.relpath(jsonl_path, REPO_ROOT)}",
        f"csv_file: {os.path.relpath(csv_path, REPO_ROOT)}",
        "",
        "Protocol",
        "  Reproduction target: FedMIA-I and FedMIA-II on AlexNet/CIFAR100.",
        "  FL method: FedAvg with model/update transfer for classification.",
        "  Target client: client 0.",
        "  Qout estimate: non-target client updates in each communication round.",
        "  FedMIA-I: negative cross-entropy loss measurement.",
        "  FedMIA-II: gradient-cosine measurement from Eq. (7).",
        "  Non-IID: Dirichlet beta grid; iid means beta=infinity.",
        "  Metrics: AUC, F1 at delta, TPR/TNR/FPR, and TPR@FPR=0.1%.",
        "",
        "Data",
        f"  data_dir: {args.data_dir}",
        f"  synthetic_data: {args.synthetic_data}",
        f"  train_samples: {data_summary['train_samples']}",
        f"  test_samples: {data_summary['test_samples']}",
        f"  train_max_samples_per_client_5: {data_summary.get('max_per_client_5', 'n/a')}",
        f"  train_max_samples_per_client_10: {data_summary.get('max_per_client_10', 'n/a')}",
        "",
        "Configuration",
        f"  runs_per_config: {args.runs}",
        f"  client_grid: {args.client_grid}",
        f"  round_grid: {args.round_grid}",
        f"  local_epoch_grid: {args.local_epoch_grid}",
        f"  beta_grid: {args.beta_grid}",
        f"  sample_fraction_grid: {args.sample_fraction_grid}",
        f"  samples_per_client_grid: {args.samples_per_client_grid}",
        f"  candidate_count: {args.candidate_count}",
        f"  attack_every: {args.attack_every}",
        f"  lr: {args.lr}",
        f"  lr_decay: {args.lr_decay}",
        f"  threshold_delta: {args.threshold}",
        f"  total_elapsed_seconds: {elapsed:.2f}",
        "",
        "Sample-Count Translation",
        "  CIFAR100 has 50,000 training images.",
        "  With SAMPLE_FRACTION_GRID=1.0 and auto samples/client:",
        "  5 clients use 10,000 train samples/client.",
        "  10 clients use 5,000 train samples/client.",
        "  This matches the paper's CIFAR100 table for 10 clients and its stated 1,000-10,000 range.",
        "",
        "Grid Results",
    ]
    for row in rows:
        lines.extend(
            [
                (
                    f"  config={int(row['config_id']):03d} clients={int(row['clients'])} "
                    f"rounds={int(row['rounds'])} local_epochs={int(row['local_epochs'])} "
                    f"beta={row['beta_label']} samples_per_client={int(row['samples_per_client'])}"
                ),
                (
                    f"    utility train_acc={row['train_acc_final_mean']:.6f} +/- {row['train_acc_final_std']:.6f} "
                    f"test_acc={row['test_acc_final_mean']:.6f} +/- {row['test_acc_final_std']:.6f}"
                ),
                (
                    f"    FedMIA-I loss auc={row['fedmia_i_loss_auc_mean']:.6f} +/- {row['fedmia_i_loss_auc_std']:.6f} "
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
                    f"    FedMIA-II cosine auc={row['fedmia_ii_cosine_auc_mean']:.6f} +/- {row['fedmia_ii_cosine_auc_std']:.6f} "
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
            ]
        )
    lines.extend(
        [
            "",
            "Commentary",
            "  This script is intentionally FedMIA-only: no LiRA shadow protocol and no BINN-specific logic.",
            "  Candidate count controls attack evaluation cost; training samples/client controls FL training volume.",
        ]
    )
    rendered = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(rendered)


def main():
    args = parse_args()
    experiment_id = make_experiment_id()
    log_path, jsonl_path = configure_logging(args, experiment_id)
    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, f"{experiment_id}.txt")
    csv_path = os.path.join(args.report_dir, f"{experiment_id}.csv")
    start_time = time.time()

    LOGGER.info("experiment_id=%s", experiment_id)
    LOGGER.info("args=%s", vars(args))
    device = resolve_device(args)
    set_seed(args.seed)
    train_x, train_y, test_x, test_y = load_cifar100(args)
    configs = make_grid(args, len(train_y))
    if not configs:
        raise ValueError("Grid is empty; check client/sample settings.")
    data_summary = {
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "max_per_client_5": len(train_y) // 5,
        "max_per_client_10": len(train_y) // 10,
    }
    LOGGER.info(
        "loaded CIFAR100 train=%s test=%s configs=%s device=%s",
        len(train_y),
        len(test_y),
        len(configs),
        device,
    )
    append_jsonl(
        jsonl_path,
        {
            "event": "experiment_start",
            "experiment_id": experiment_id,
            "args": vars(args),
            "data_summary": data_summary,
            "configs": [config.__dict__ for config in configs],
        },
    )

    rows = []
    global_run_id = 1
    for config in configs:
        LOGGER.info(
            "config=%03d start clients=%s rounds=%s local_epochs=%s beta=%s samples_per_client=%s runs=%s",
            config.config_id,
            config.clients,
            config.rounds,
            config.local_epochs,
            config.beta_label,
            config.samples_per_client,
            args.runs,
        )
        results = []
        for run_offset in range(args.runs):
            run_seed = args.seed + config.config_id * 100_000 + run_offset * 1_009
            result = run_one_trajectory(
                config,
                global_run_id,
                run_seed,
                train_x,
                train_y,
                test_x,
                test_y,
                args,
                device,
                jsonl_path,
            )
            results.append(result)
            global_run_id += 1
        summary = summarize_config(results, args)
        row = {
            "config_id": config.config_id,
            "clients": config.clients,
            "rounds": config.rounds,
            "local_epochs": config.local_epochs,
            "beta_label": config.beta_label,
            "sample_fraction": config.sample_fraction if config.sample_fraction is not None else "",
            "samples_per_client": config.samples_per_client,
            **summary,
        }
        rows.append(row)
        write_csv(csv_path, rows)
        append_jsonl(jsonl_path, {"event": "config_result", **row})
        LOGGER.info(
            "config=%03d result fedmia_i_auc=%.4f fedmia_ii_auc=%.4f test_acc=%.4f",
            config.config_id,
            row["fedmia_i_loss_auc_mean"],
            row["fedmia_ii_cosine_auc_mean"],
            row["test_acc_final_mean"],
        )

    elapsed = time.time() - start_time
    write_report(report_path, experiment_id, log_path, jsonl_path, csv_path, args, data_summary, rows, elapsed)
    LOGGER.info("report_path=%s", report_path)


if __name__ == "__main__":
    main()
