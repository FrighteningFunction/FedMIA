import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from attacks.fedmia import (
    FedMIAConfig,
    FedMIARoundInputs,
    FedMIARoundMeasurements,
    build_measurement_rounds,
    cosine_measurement,
    evaluate_membership,
)


def build_round(
    target_values,
    reference_base,
    reference_offsets,
    round_id,
):
    references = []
    for offset in reference_offsets:
        references.append([base + offset for base in reference_base])
    return FedMIARoundMeasurements(
        target_measurements=target_values,
        reference_measurements=references,
        round_id=round_id,
    )


def main():
    # Five candidate member samples measured across three communication rounds.
    member_rounds = [
        build_round(
            target_values=[0.84, 0.79, 0.88, 0.82, 0.86],
            reference_base=[0.18, 0.12, 0.20, 0.14, 0.17],
            reference_offsets=[-0.02, 0.00, 0.01, 0.03],
            round_id=1,
        ),
        build_round(
            target_values=[0.81, 0.77, 0.85, 0.80, 0.83],
            reference_base=[0.16, 0.11, 0.19, 0.13, 0.15],
            reference_offsets=[-0.03, -0.01, 0.02, 0.04],
            round_id=2,
        ),
        build_round(
            target_values=[0.83, 0.78, 0.87, 0.81, 0.84],
            reference_base=[0.17, 0.10, 0.18, 0.12, 0.16],
            reference_offsets=[-0.01, 0.00, 0.02, 0.05],
            round_id=3,
        ),
    ]

    # The same number of non-member candidate samples, where target scores stay near Q_out.
    nonmember_rounds = [
        build_round(
            target_values=[0.15, 0.10, 0.22, 0.11, 0.16],
            reference_base=[0.18, 0.12, 0.20, 0.14, 0.17],
            reference_offsets=[-0.02, 0.00, 0.01, 0.03],
            round_id=1,
        ),
        build_round(
            target_values=[0.14, 0.09, 0.18, 0.12, 0.14],
            reference_base=[0.16, 0.11, 0.19, 0.13, 0.15],
            reference_offsets=[-0.03, -0.01, 0.02, 0.04],
            round_id=2,
        ),
        build_round(
            target_values=[0.16, 0.11, 0.20, 0.13, 0.15],
            reference_base=[0.17, 0.10, 0.18, 0.12, 0.16],
            reference_offsets=[-0.01, 0.00, 0.02, 0.05],
            round_id=3,
        ),
    ]

    evaluation = evaluate_membership(
        member_rounds=member_rounds,
        nonmember_rounds=nonmember_rounds,
        config=FedMIAConfig(threshold=0.5),
    )

    member_mean = sum(evaluation.member_scores.aggregate_scores) / len(evaluation.member_scores.aggregate_scores)
    nonmember_mean = sum(evaluation.nonmember_scores.aggregate_scores) / len(evaluation.nonmember_scores.aggregate_scores)

    print("FedMIA portable smoke test")
    print("member aggregate scores:", [round(score, 6) for score in evaluation.member_scores.aggregate_scores])
    print("nonmember aggregate scores:", [round(score, 6) for score in evaluation.nonmember_scores.aggregate_scores])
    print("member mean:", round(member_mean, 6))
    print("nonmember mean:", round(nonmember_mean, 6))
    print("auc:", round(evaluation.auc, 6))
    print("log_auc:", round(evaluation.log_auc, 6))
    print("tpr@0.001:", round(evaluation.tprs["0.001"], 6))

    assert member_mean > 0.95, "Expected members to score strongly above the null distribution."
    assert nonmember_mean < 0.5, "Expected non-members to remain near or below the null distribution."
    assert evaluation.auc > 0.95, "Expected the synthetic smoke test to be clearly separable."
    assert all(pred == 1 for pred in evaluation.member_scores.predictions), "All synthetic members should be predicted as members."
    assert any(pred == 0 for pred in evaluation.nonmember_scores.predictions), "At least one synthetic non-member should remain below threshold."

    # A second smoke path through the callback interface: generic FL systems only
    # need to supply round updates and a measurement callback.
    round_inputs = [
        FedMIARoundInputs(
            target_update=[1.0, 0.9, 0.1],
            reference_updates=[
                [0.1, 0.1, 0.9],
                [0.2, 0.1, 0.8],
                [0.1, 0.2, 0.7],
            ],
            round_id=1,
        ),
        FedMIARoundInputs(
            target_update=[0.9, 1.0, 0.1],
            reference_updates=[
                [0.1, 0.2, 0.8],
                [0.2, 0.2, 0.7],
                [0.1, 0.1, 0.9],
            ],
            round_id=2,
        ),
    ]
    member_samples = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.1]]
    nonmember_samples = [[0.1, 0.1, 0.9], [0.2, 0.1, 0.8]]

    member_rounds_via_adapter = build_measurement_rounds(
        round_inputs=round_inputs,
        candidate_samples=member_samples,
        measurement_fn=lambda update, sample, _: cosine_measurement(update, sample),
    )
    nonmember_rounds_via_adapter = build_measurement_rounds(
        round_inputs=round_inputs,
        candidate_samples=nonmember_samples,
        measurement_fn=lambda update, sample, _: cosine_measurement(update, sample),
    )
    adapter_evaluation = evaluate_membership(
        member_rounds=member_rounds_via_adapter,
        nonmember_rounds=nonmember_rounds_via_adapter,
        config=FedMIAConfig(threshold=0.5),
    )
    print("adapter auc:", round(adapter_evaluation.auc, 6))
    assert adapter_evaluation.auc > 0.95, "Expected the adapter-based smoke path to separate members and non-members."


if __name__ == "__main__":
    main()
