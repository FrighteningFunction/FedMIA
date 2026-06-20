import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import (
    FedMIAConfig,
    FedMIARoundMeasurements,
    evaluate_membership,
    normal_cdf,
    score_round,
)
from attacks.repo_adapter import evaluate_artifact_series


class FedMIAMathTests(unittest.TestCase):
    def test_normal_cdf_is_half_at_mean_and_monotonic(self):
        self.assertAlmostEqual(normal_cdf(0.0, 0.0, 1.0), 0.5, places=7)
        self.assertGreater(normal_cdf(2.0, 0.0, 1.0), normal_cdf(1.0, 0.0, 1.0))
        self.assertGreater(normal_cdf(1.0, 0.0, 1.0), normal_cdf(-1.0, 0.0, 1.0))

    def test_round_scoring_produces_sensible_probabilities(self):
        round_measurements = FedMIARoundMeasurements(
            target_measurements=[0.2, 0.5, 0.8],
            reference_measurements=[
                [0.1, 0.2, 0.3],
                [0.1, 0.25, 0.35],
                [0.15, 0.3, 0.4],
                [0.05, 0.2, 0.45],
            ],
            round_id=1,
        )
        scores, mu_out, var_out = score_round(round_measurements, FedMIAConfig())

        self.assertEqual(len(scores), 3)
        self.assertTrue(0.0 <= scores[0] <= 1.0)
        self.assertTrue(0.0 <= scores[1] <= 1.0)
        self.assertTrue(0.0 <= scores[2] <= 1.0)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[1], scores[0])
        self.assertAlmostEqual(mu_out[0], 0.1, places=6)
        self.assertGreaterEqual(var_out[0], 1e-8)


class FedMIAIntegrationTests(unittest.TestCase):
    def _build_round(self, target_values, reference_base, reference_offsets, round_id):
        references = []
        for offset in reference_offsets:
            references.append([base + offset for base in reference_base])
        return FedMIARoundMeasurements(
            target_measurements=target_values,
            reference_measurements=references,
            round_id=round_id,
        )

    def test_portable_fedmia_separates_member_and_nonmember(self):
        member_rounds = [
            self._build_round([0.84, 0.79, 0.88], [0.18, 0.12, 0.20], [-0.02, 0.00, 0.03], 1),
            self._build_round([0.81, 0.77, 0.85], [0.16, 0.11, 0.19], [-0.03, -0.01, 0.04], 2),
            self._build_round([0.83, 0.78, 0.87], [0.17, 0.10, 0.18], [-0.01, 0.00, 0.05], 3),
        ]
        nonmember_rounds = [
            self._build_round([0.15, 0.10, 0.22], [0.18, 0.12, 0.20], [-0.02, 0.00, 0.03], 1),
            self._build_round([0.14, 0.09, 0.18], [0.16, 0.11, 0.19], [-0.03, -0.01, 0.04], 2),
            self._build_round([0.16, 0.11, 0.20], [0.17, 0.10, 0.18], [-0.01, 0.00, 0.05], 3),
        ]
        evaluation = evaluate_membership(member_rounds, nonmember_rounds, FedMIAConfig(threshold=0.5))

        member_mean = sum(evaluation.member_scores.aggregate_scores) / len(evaluation.member_scores.aggregate_scores)
        nonmember_mean = sum(evaluation.nonmember_scores.aggregate_scores) / len(evaluation.nonmember_scores.aggregate_scores)

        self.assertGreater(member_mean, 0.95)
        self.assertLess(nonmember_mean, 0.5)
        self.assertGreater(evaluation.auc, 0.95)
        self.assertEqual(evaluation.tprs["0.001"], 1.0)

    def test_repo_artifact_adapter_produces_expected_separation(self):
        def logits_from_losses(losses):
            return torch.tensor([[-float(loss)] for loss in losses], dtype=torch.float32)

        def labels(count):
            return torch.ones(count, 1, dtype=torch.float32)

        training_res = [
            {
                "test_acc": 1.0,
                "tarin_cos": torch.tensor([0.9, 0.85, 0.88]),
                "test_cos": torch.tensor([0.1, 0.12, 0.11]),
                "mix_cos": torch.tensor([0.18, 0.21, 0.2]),
                "train_res": {"logit": logits_from_losses([0.08, 0.09, 0.1]), "labels": labels(3)},
                "test_res": {"logit": logits_from_losses([0.7, 0.65, 0.75]), "labels": labels(3)},
                "mix_res": {"logit": logits_from_losses([0.8, 0.78, 0.82]), "labels": labels(3)},
                "val_res": {"logit": logits_from_losses([0.68, 0.71, 0.69]), "labels": labels(3)},
            },
            {
                "test_acc": 1.0,
                "tarin_cos": torch.tensor([0.2, 0.18, 0.24]),
                "test_cos": torch.tensor([0.22, 0.2, 0.21]),
                "mix_cos": torch.tensor([0.23, 0.21, 0.22]),
                "train_res": {"logit": logits_from_losses([0.72, 0.74, 0.7]), "labels": labels(3)},
                "test_res": {"logit": logits_from_losses([0.69, 0.7, 0.73]), "labels": labels(3)},
                "mix_res": {"logit": logits_from_losses([0.77, 0.79, 0.8]), "labels": labels(3)},
                "val_res": {"logit": logits_from_losses([0.71, 0.72, 0.74]), "labels": labels(3)},
            },
            {
                "test_acc": 1.0,
                "tarin_cos": torch.tensor([0.19, 0.21, 0.2]),
                "test_cos": torch.tensor([0.18, 0.2, 0.22]),
                "mix_cos": torch.tensor([0.19, 0.2, 0.18]),
                "train_res": {"logit": logits_from_losses([0.75, 0.73, 0.76]), "labels": labels(3)},
                "test_res": {"logit": logits_from_losses([0.7, 0.72, 0.69]), "labels": labels(3)},
                "mix_res": {"logit": logits_from_losses([0.81, 0.82, 0.8]), "labels": labels(3)},
                "val_res": {"logit": logits_from_losses([0.72, 0.75, 0.71]), "labels": labels(3)},
            },
        ]

        evaluation = evaluate_artifact_series(
            epoch_training_results=[training_res],
            attack_mode="cos",
            mode="test",
            config=FedMIAConfig(threshold=0.5),
        )

        member_scores = evaluation.member_scores.aggregate_scores
        nonmember_scores = evaluation.nonmember_scores.aggregate_scores

        self.assertTrue(all(score > 0.99 for score in member_scores))
        self.assertTrue(all(score < 0.5 for score in nonmember_scores))
        self.assertGreater(evaluation.auc, 0.95)


if __name__ == "__main__":
    unittest.main()
