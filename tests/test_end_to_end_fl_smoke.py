import copy
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import (
    FedMIAConfig,
    FedMIARoundInputs,
    build_measurement_rounds,
    evaluate_membership,
)
from experiments.trainer_private import TrainerPrivate
from utils.federated import fed_avg_state_dicts


class TinyBinaryNet(torch.nn.Module):
    task_type = "binary"

    def __init__(self, input_dim=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_client_dataset(client_id, samples_per_class=12, scale=0.35, target_client=False):
    generator = torch.Generator().manual_seed(100 + client_id)
    if target_client:
        neg_center = torch.tensor([-1.9, -1.7], dtype=torch.float32)
        pos_center = torch.tensor([1.9, 1.7], dtype=torch.float32)
    else:
        neg_center = torch.tensor([-1.2, -0.9], dtype=torch.float32) + 0.2 * client_id
        pos_center = torch.tensor([1.0, 0.9], dtype=torch.float32) + 0.2 * client_id
    neg = neg_center + scale * torch.randn(samples_per_class, 2, generator=generator)
    pos = pos_center + scale * torch.randn(samples_per_class, 2, generator=generator)
    x = torch.cat([neg, pos], dim=0)
    y = torch.cat([
        torch.zeros(samples_per_class, 1, dtype=torch.float32),
        torch.ones(samples_per_class, 1, dtype=torch.float32),
    ], dim=0)
    return TensorDataset(x, y)


def make_holdout_dataset(samples_per_class=12):
    generator = torch.Generator().manual_seed(999)
    neg = torch.tensor([-0.7, -0.4], dtype=torch.float32) + 0.45 * torch.randn(samples_per_class, 2, generator=generator)
    pos = torch.tensor([0.7, 0.4], dtype=torch.float32) + 0.45 * torch.randn(samples_per_class, 2, generator=generator)
    x = torch.cat([neg, pos], dim=0)
    y = torch.cat([
        torch.zeros(samples_per_class, 1, dtype=torch.float32),
        torch.ones(samples_per_class, 1, dtype=torch.float32),
    ], dim=0)
    return TensorDataset(x, y)


def binary_accuracy(model, dataset):
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs)
            preds = (torch.sigmoid(logits).view(-1) >= 0.5).float()
            truth = labels.view(-1)
            correct += preds.eq(truth).sum().item()
            total += truth.numel()
    return correct / total


def model_update(global_state, local_state):
    return {key: global_state[key] - local_state[key] for key in global_state.keys()}


def sample_gradient_dict(model, sample):
    x, y = sample
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 0:
        y = y.view(1, 1)
    elif y.dim() == 1:
        y = y.unsqueeze(0)

    model.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.view_as(logits))
    grads = torch.autograd.grad(loss, [param for param in model.parameters() if param.requires_grad], allow_unused=True)

    grad_dict = {}
    grad_iter = iter(grads)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = next(grad_iter)
        grad_dict[name] = torch.zeros_like(param) if grad is None else grad.detach().clone()
    return grad_dict


def negative_sample_loss(model, sample):
    x, y = sample
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 0:
        y = y.view(1, 1)
    elif y.dim() == 1:
        y = y.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.view_as(logits), reduction="mean")
    return -float(loss.item())


class EndToEndFederatedSmokeTest(unittest.TestCase):
    def test_federated_training_and_portable_fedmia_work_together(self):
        torch.manual_seed(123)
        device = torch.device("cpu")

        num_clients = 3
        rounds = 3
        local_epochs = 10
        learning_rate = 0.12

        client_datasets = [
            make_client_dataset(0, samples_per_class=3, scale=0.12, target_client=True),
            make_client_dataset(1, samples_per_class=12, scale=0.35),
            make_client_dataset(2, samples_per_class=12, scale=0.35),
        ]
        client_loaders = [DataLoader(dataset, batch_size=8, shuffle=True) for dataset in client_datasets]
        holdout_dataset = make_holdout_dataset()
        combined_dataset = TensorDataset(
            torch.cat([dataset.tensors[0] for dataset in client_datasets], dim=0),
            torch.cat([dataset.tensors[1] for dataset in client_datasets], dim=0),
        )

        global_model = TinyBinaryNet().to(device)
        initial_acc = binary_accuracy(global_model, combined_dataset)

        member_samples = [client_datasets[0][idx] for idx in range(3, len(client_datasets[0]))]
        nonmember_samples = [holdout_dataset[idx] for idx in range(12, 18)]
        round_inputs = []

        for round_id in range(rounds):
            global_state = copy.deepcopy(global_model.state_dict())
            local_states = []
            local_models = []

            for client_loader in client_loaders:
                local_model = TinyBinaryNet().to(device)
                local_model.load_state_dict(global_state)
                trainer = TrainerPrivate(
                    model=local_model,
                    train_set=client_loader,
                    device=device,
                    dp=False,
                    sigma=0.0,
                    num_classes=1,
                    defense="none",
                )
                local_state, _ = trainer._local_update_noback(
                    dataloader=client_loader,
                    local_ep=local_epochs,
                    lr=learning_rate,
                    optim_choice="sgd",
                    sampling_proportion=1.0,
                )
                local_states.append(copy.deepcopy(local_state))
                trained_model = TinyBinaryNet().to(device)
                trained_model.load_state_dict(local_state)
                local_models.append(trained_model)

            averaged_state = fed_avg_state_dicts(local_states, [1.0] * num_clients)
            global_model.load_state_dict(averaged_state)
            round_inputs.append(
                FedMIARoundInputs(
                    target_update=local_models[0],
                    reference_updates=local_models[1:],
                    global_model=None,
                    round_id=round_id,
                )
            )

        final_acc = binary_accuracy(global_model, combined_dataset)
        self.assertGreater(final_acc, initial_acc)
        self.assertGreaterEqual(final_acc, 0.8)

        member_rounds = build_measurement_rounds(
            round_inputs=round_inputs,
            candidate_samples=member_samples,
            measurement_fn=lambda model_obj, sample, _: negative_sample_loss(model_obj, sample),
        )
        nonmember_rounds = build_measurement_rounds(
            round_inputs=round_inputs,
            candidate_samples=nonmember_samples,
            measurement_fn=lambda model_obj, sample, _: negative_sample_loss(model_obj, sample),
        )

        evaluation = evaluate_membership(
            member_rounds=member_rounds,
            nonmember_rounds=nonmember_rounds,
            config=FedMIAConfig(threshold=0.005),
        )

        member_mean = sum(evaluation.member_scores.aggregate_scores) / len(evaluation.member_scores.aggregate_scores)
        nonmember_mean = sum(evaluation.nonmember_scores.aggregate_scores) / len(evaluation.nonmember_scores.aggregate_scores)

        print("initial_acc", round(initial_acc, 6))
        print("final_acc", round(final_acc, 6))
        print("member_scores", [round(score, 6) for score in evaluation.member_scores.aggregate_scores])
        print("nonmember_scores", [round(score, 6) for score in evaluation.nonmember_scores.aggregate_scores])
        print("fedmia_auc", round(evaluation.auc, 6))

        self.assertGreater(member_mean, nonmember_mean)
        self.assertGreaterEqual(evaluation.auc, 0.66)
        self.assertGreater(sum(evaluation.member_scores.predictions), 0)
        self.assertGreater(sum(1 - pred for pred in evaluation.nonmember_scores.predictions), 0)


if __name__ == "__main__":
    unittest.main()
