from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from .fedmia import FedMIAConfig, FedMIAEvaluation, FedMIARoundMeasurements, evaluate_membership


def _negative_losses(logits, labels):
    """
    Repo artifact adapter for both multiclass CE and binary BCE logits.

    Returns a measurement where larger means "more member-like", matching the
    FedMIA assumption used elsewhere in the repository.
    """
    logits = logits.detach().cpu() if hasattr(logits, "detach") else logits
    labels = labels.detach().cpu() if hasattr(labels, "detach") else labels

    if hasattr(logits, "shape") and len(logits.shape) == 2 and logits.shape[1] == 1:
        import torch
        import torch.nn.functional as F

        losses = F.binary_cross_entropy_with_logits(
            logits,
            labels.float().view_as(logits),
            reduction="none",
        ).view(-1)
        return [-float(value) for value in losses.tolist()]

    import torch
    import torch.nn.functional as F

    losses = F.cross_entropy(logits, labels.long(), reduction="none")
    return [-float(value) for value in losses.tolist()]


def _sample_test_measurements(target_res: Dict[str, Any], mode: str, attack_mode: str, mix_length: int | None):
    if attack_mode == "cos":
        if mode == "test":
            return [float(value) for value in target_res["test_cos"].detach().cpu().tolist()]
        if mode == "val":
            return [float(value) for value in target_res["val_cos"].detach().cpu().tolist()]
        if mode == "mix":
            test_values = [float(value) for value in target_res["test_cos"][:mix_length].detach().cpu().tolist()]
            mix_values = [float(value) for value in target_res["mix_cos"].detach().cpu().tolist()]
            return test_values + mix_values

    if attack_mode == "loss":
        if mode == "test":
            return _negative_losses(target_res["test_res"]["logit"], target_res["test_res"]["labels"])
        if mode == "val":
            return _negative_losses(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
        if mode == "mix":
            test_logits = target_res["test_res"]["logit"][:mix_length]
            test_labels = target_res["test_res"]["labels"][:mix_length]
            return _negative_losses(test_logits, test_labels) + _negative_losses(
                target_res["mix_res"]["logit"],
                target_res["mix_res"]["labels"],
            )

    raise ValueError(f"Unsupported mode={mode} or attack_mode={attack_mode}")


def _shadow_test_measurements(shadow_res: Dict[str, Any], mode: str, attack_mode: str, mix_length: int | None):
    return _sample_test_measurements(shadow_res, mode, attack_mode, mix_length)


def artifact_round_to_portable(
    training_res: Sequence[Dict[str, Any]],
    attack_mode: str = "cos",
    mode: str = "test",
    mix_length: int | None = None,
    round_id: int | None = None,
) -> Tuple[FedMIARoundMeasurements, FedMIARoundMeasurements]:
    target_res = training_res[0]
    shadow_res = training_res[1:]

    if attack_mode == "cos":
        target_member = [float(value) for value in target_res["tarin_cos"].detach().cpu().tolist()]
        shadow_member = [
            [float(value) for value in shadow["tarin_cos"].detach().cpu().tolist()]
            for shadow in shadow_res
        ]
    elif attack_mode == "loss":
        target_member = _negative_losses(target_res["train_res"]["logit"], target_res["train_res"]["labels"])
        shadow_member = [
            _negative_losses(shadow["train_res"]["logit"], shadow["train_res"]["labels"])
            for shadow in shadow_res
        ]
    else:
        raise ValueError(f"Unsupported attack_mode={attack_mode}")

    target_nonmember = _sample_test_measurements(target_res, mode, attack_mode, mix_length)
    shadow_nonmember = [
        _shadow_test_measurements(shadow, mode, attack_mode, mix_length)
        for shadow in shadow_res
    ]

    member_round = FedMIARoundMeasurements(
        target_measurements=target_member,
        reference_measurements=shadow_member,
        round_id=round_id,
        metadata={"split": "member", "attack_mode": attack_mode},
    )
    nonmember_round = FedMIARoundMeasurements(
        target_measurements=target_nonmember,
        reference_measurements=shadow_nonmember,
        round_id=round_id,
        metadata={"split": "nonmember", "attack_mode": attack_mode},
    )
    return member_round, nonmember_round


def evaluate_artifact_series(
    epoch_training_results: Sequence[Sequence[Dict[str, Any]]],
    attack_mode: str = "cos",
    mode: str = "test",
    mix_length: int | None = None,
    config: FedMIAConfig | None = None,
) -> FedMIAEvaluation:
    member_rounds: List[FedMIARoundMeasurements] = []
    nonmember_rounds: List[FedMIARoundMeasurements] = []

    for round_index, training_res in enumerate(epoch_training_results):
        member_round, nonmember_round = artifact_round_to_portable(
            training_res=training_res,
            attack_mode=attack_mode,
            mode=mode,
            mix_length=mix_length,
            round_id=round_index,
        )
        member_rounds.append(member_round)
        nonmember_rounds.append(nonmember_round)

    return evaluate_membership(member_rounds, nonmember_rounds, config=config)
