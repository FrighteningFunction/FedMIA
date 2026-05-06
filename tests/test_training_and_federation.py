import copy
import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


def binary_accuracy(model, loader):
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


class TrainingLoopTests(unittest.TestCase):
    def _make_separable_loader(self):
        inputs = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.5, -1.2],
                [-1.0, -0.5],
                [-0.8, -1.3],
                [1.2, 1.0],
                [1.5, 1.1],
                [2.0, 1.8],
                [1.0, 1.4],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset, batch_size=4, shuffle=True)

    def test_local_binary_training_improves_accuracy_and_changes_weights(self):
        torch.manual_seed(7)
        model = TinyBinaryNet()
        loader = self._make_separable_loader()
        trainer = TrainerPrivate(
            model=model,
            train_set=loader,
            device=torch.device("cpu"),
            dp=False,
            sigma=0.0,
            num_classes=1,
            defense="none",
        )

        before_state = copy.deepcopy(model.state_dict())
        before_acc = binary_accuracy(model, loader)

        trainer._local_update_noback(
            dataloader=loader,
            local_ep=25,
            lr=0.1,
            optim_choice="sgd",
            sampling_proportion=1.0,
        )

        after_acc = binary_accuracy(model, loader)
        self.assertGreater(after_acc, before_acc)
        self.assertGreaterEqual(after_acc, 0.875)

        any_changed = any(
            not torch.allclose(before_state[key], model.state_dict()[key])
            for key in before_state
        )
        self.assertTrue(any_changed, "Expected at least one parameter tensor to change after local training.")


class FederatedAggregationTests(unittest.TestCase):
    def test_fedavg_matches_expected_weighted_average(self):
        state_a = {
            "w": torch.tensor([1.0, 3.0]),
            "b": torch.tensor([2.0]),
        }
        state_b = {
            "w": torch.tensor([5.0, 7.0]),
            "b": torch.tensor([6.0]),
        }
        state_c = {
            "w": torch.tensor([9.0, 11.0]),
            "b": torch.tensor([10.0]),
        }
        averaged = fed_avg_state_dicts([state_a, state_b, state_c], [1, 2, 1])

        expected_w = (state_a["w"] * 1 + state_b["w"] * 2 + state_c["w"] * 1) / 4.0
        expected_b = (state_a["b"] * 1 + state_b["b"] * 2 + state_c["b"] * 1) / 4.0

        self.assertTrue(torch.allclose(averaged["w"], expected_w))
        self.assertTrue(torch.allclose(averaged["b"], expected_b))

    def test_fedavg_normalizes_weights(self):
        state_a = {"w": torch.tensor([2.0])}
        state_b = {"w": torch.tensor([6.0])}
        averaged = fed_avg_state_dicts([state_a, state_b], [0.25, 0.75])
        self.assertAlmostEqual(averaged["w"].item(), 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
