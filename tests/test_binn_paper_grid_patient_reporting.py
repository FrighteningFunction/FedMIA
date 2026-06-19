import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments import fedmia_binn_paper_grid as paper_grid


class BINNPaperGridCandidateTests(unittest.TestCase):
    def test_target_nonmembers_source_includes_holdout_and_non_target_clients(self):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
        client_indices = [[0, 1], [2, 3, 4], [5, 6]]
        holdout_indices = [7, 8]

        nonmember_pool = paper_grid.nonmember_pool_for_source(
            client_indices,
            holdout_indices,
            "target_nonmembers",
        )

        self.assertEqual(set(nonmember_pool), {2, 3, 4, 5, 6, 7, 8})
        self.assertTrue(set(nonmember_pool).isdisjoint({0, 1}))

    def test_other_clients_nonmember_source_excludes_holdout_and_target(self):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
        client_indices = [[0, 1], [2, 3, 4], [5, 6]]
        holdout_indices = [7, 8]
        rng = np.random.default_rng(123)

        members, nonmembers = paper_grid.make_candidate_indices(
            labels,
            client_indices,
            holdout_indices,
            candidate_count=0,
            rng=rng,
            nonmember_source="other_clients",
        )

        self.assertEqual(len(members), 2)
        self.assertEqual(len(nonmembers), 2)
        self.assertTrue(set(members).issubset({0, 1}))
        self.assertTrue(set(nonmembers).issubset({2, 3, 4, 5, 6}))
        self.assertTrue(set(nonmembers).isdisjoint(holdout_indices))

    def test_holdout_nonmember_source_preserves_old_behavior(self):
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
        client_indices = [[0, 1], [2, 3, 4], [5, 6]]
        holdout_indices = [7, 8]
        rng = np.random.default_rng(123)

        members, nonmembers = paper_grid.make_candidate_indices(
            labels,
            client_indices,
            holdout_indices,
            candidate_count=0,
            rng=rng,
            nonmember_source="holdout",
        )

        self.assertEqual(len(members), 2)
        self.assertEqual(len(nonmembers), 2)
        self.assertTrue(set(nonmembers).issubset(set(holdout_indices)))

    def test_audited_patients_are_forced_into_both_states_across_runs(self):
        client_indices = [[0, 1, 2], [3, 4], [5, 6]]
        audit_indices = [0, 3, 5, 7]

        clients_run_1, members_1, nonmembers_1 = paper_grid.apply_audit_patient_assignments(
            client_indices,
            audit_indices,
            run_id=1,
            rng=np.random.default_rng(123),
        )
        clients_run_2, members_2, nonmembers_2 = paper_grid.apply_audit_patient_assignments(
            client_indices,
            audit_indices,
            run_id=2,
            rng=np.random.default_rng(456),
        )

        self.assertEqual(set(members_1), set(nonmembers_2))
        self.assertEqual(set(nonmembers_1), set(members_2))
        self.assertTrue(set(members_1).issubset(set(clients_run_1[0])))
        self.assertTrue(set(members_2).issubset(set(clients_run_2[0])))
        self.assertTrue(set(nonmembers_1).isdisjoint(set(clients_run_1[0])))
        self.assertTrue(set(nonmembers_2).isdisjoint(set(clients_run_2[0])))

    def test_audited_holdout_out_state_does_not_require_two_clients(self):
        client_indices = [[0, 1, 2, 3]]
        audit_indices = [0, 1]

        clients, members, nonmembers = paper_grid.apply_audit_patient_assignments(
            client_indices,
            audit_indices,
            run_id=1,
            rng=np.random.default_rng(123),
            nonmember_source="holdout",
        )

        self.assertEqual(len(clients), 1)
        self.assertEqual(set(members) | set(nonmembers), set(audit_indices))
        self.assertTrue(set(members).issubset(set(clients[0])))
        self.assertTrue(set(nonmembers).isdisjoint(set(clients[0])))


class BINNPatientVulnerabilityTests(unittest.TestCase):
    @staticmethod
    def observation(patient, label, scores, membership_label):
        rows = []
        for run_id, score in enumerate(scores, start=1):
            prediction = 1 if score > 0.5 else 0
            rows.append(
                {
                    "config_id": 1,
                    "run": run_id,
                    "seed": 100 + run_id,
                    "measurement": "fedmia_i_loss",
                    "nonmember_source": "target_nonmembers",
                    "patient_index": patient,
                    "class_label": label,
                    "membership_label": membership_label,
                    "prediction": prediction,
                    "correct": 1 if prediction == membership_label else 0,
                    "score": score,
                    "auc_contribution": 0.5,
                }
            )
        return rows

    def test_patient_vulnerability_summary_requires_both_states(self):
        observations = []
        observations.extend(self.observation(10, 1, [0.9, 0.8], 1))
        observations.extend(self.observation(10, 1, [0.2, 0.1], 0))
        observations.extend(self.observation(11, 0, [0.2, 0.3], 1))
        observations.extend(self.observation(11, 0, [0.8, 0.7], 0))
        observations.extend(self.observation(12, 1, [0.95, 0.91], 1))

        patient_rows = paper_grid.make_patient_metric_rows(observations)
        summary = paper_grid.summarize_patient_vulnerability(
            patient_rows,
            min_state_appearances=2,
        )

        self.assertEqual(summary["fedmia_i_loss_patient_eligible_patients"], 2.0)
        self.assertEqual(summary["fedmia_i_loss_patient_most_vulnerable_patient"], 10.0)
        self.assertEqual(summary["fedmia_i_loss_patient_least_vulnerable_patient"], 11.0)
        self.assertAlmostEqual(summary["fedmia_i_loss_patient_most_vulnerable_auc"], 1.0)
        self.assertAlmostEqual(summary["fedmia_i_loss_patient_least_vulnerable_auc"], 0.0)
        self.assertTrue(math.isnan(summary["fedmia_ii_cosine_patient_auc_mean"]))


if __name__ == "__main__":
    unittest.main()
