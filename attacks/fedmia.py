from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Number = float
MeasurementFunction = Callable[[Any, Any, Any], float]


def _is_sequence_like(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _mean(values: Sequence[Number]) -> Number:
    return sum(values) / max(len(values), 1)


def _variance(values: Sequence[Number], mean_value: Optional[Number] = None) -> Number:
    if not values:
        return 0.0
    if mean_value is None:
        mean_value = _mean(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def _std(values: Sequence[Number], mean_value: Optional[Number] = None) -> Number:
    return math.sqrt(_variance(values, mean_value))


def normal_cdf(x: Number, mu: Number, variance: Number) -> Number:
    safe_variance = max(float(variance), 1e-12)
    z = (float(x) - float(mu)) / math.sqrt(2.0 * safe_variance)
    return 0.5 * (1.0 + math.erf(z))


def flatten_update(update: Any) -> List[float]:
    """
    Flatten nested lists/tuples/dicts or tensor-like structures into a 1D float list.

    This is the small portability hinge for the attack: Flower, plain PyTorch, or a
    custom trainer only needs to provide updates in any recursively traversable form.
    """
    update = _to_python(update)
    flat: List[float] = []

    if isinstance(update, dict):
        for key in sorted(update.keys()):
            flat.extend(flatten_update(update[key]))
        return flat

    if _is_sequence_like(update):
        for value in update:
            flat.extend(flatten_update(value))
        return flat

    flat.append(float(update))
    return flat


def cosine_measurement(update: Any, sample_gradient: Any) -> float:
    update_flat = flatten_update(update)
    gradient_flat = flatten_update(sample_gradient)
    if len(update_flat) != len(gradient_flat):
        raise ValueError(
            f"Update and gradient lengths differ: {len(update_flat)} != {len(gradient_flat)}"
        )

    dot = sum(left * right for left, right in zip(update_flat, gradient_flat))
    update_norm = math.sqrt(sum(value * value for value in update_flat))
    gradient_norm = math.sqrt(sum(value * value for value in gradient_flat))
    if update_norm == 0.0 or gradient_norm == 0.0:
        return 0.0
    return dot / (update_norm * gradient_norm)


@dataclass
class FedMIAConfig:
    threshold: float = 0.5
    outlier_std_factor: float = 3.0
    min_variance: float = 1e-8


@dataclass
class FedMIARoundInputs:
    target_update: Any
    reference_updates: Sequence[Any]
    global_model: Any = None
    round_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FedMIARoundMeasurements:
    target_measurements: List[float]
    reference_measurements: List[List[float]]
    round_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.target_measurements = [float(value) for value in _to_python(self.target_measurements)]
        self.reference_measurements = [
            [float(value) for value in _to_python(row)]
            for row in _to_python(self.reference_measurements)
        ]
        if not self.reference_measurements:
            raise ValueError("FedMIA requires at least one non-target reference update per round.")
        sample_count = len(self.target_measurements)
        for row in self.reference_measurements:
            if len(row) != sample_count:
                raise ValueError("All reference measurement rows must match target measurement count.")


@dataclass
class FedMIAScores:
    per_round_scores: List[List[float]]
    aggregate_scores: List[float]
    predictions: List[int]
    per_round_mu_out: List[List[float]]
    per_round_var_out: List[List[float]]


@dataclass
class FedMIAEvaluation:
    member_scores: FedMIAScores
    nonmember_scores: FedMIAScores
    auc: float
    log_auc: float
    tprs: Dict[str, float]


def build_measurement_rounds(
    round_inputs: Sequence[FedMIARoundInputs],
    candidate_samples: Sequence[Any],
    measurement_fn: MeasurementFunction,
) -> List[FedMIARoundMeasurements]:
    measurement_rounds: List[FedMIARoundMeasurements] = []
    for round_input in round_inputs:
        target_measurements = [
            measurement_fn(round_input.target_update, sample, round_input.global_model)
            for sample in candidate_samples
        ]
        reference_measurements = []
        for reference_update in round_input.reference_updates:
            reference_measurements.append(
                [
                    measurement_fn(reference_update, sample, round_input.global_model)
                    for sample in candidate_samples
                ]
            )
        measurement_rounds.append(
            FedMIARoundMeasurements(
                target_measurements=target_measurements,
                reference_measurements=reference_measurements,
                round_id=round_input.round_id,
                metadata=dict(round_input.metadata),
            )
        )
    return measurement_rounds


def _estimate_null_distribution(
    reference_measurements: Sequence[Sequence[float]],
    config: FedMIAConfig,
) -> Tuple[List[float], List[float]]:
    """
    Estimate the paper's Q_out distribution for every candidate sample.

    ``reference_measurements`` is shaped as:
        reference client x candidate sample

    For each candidate sample, FedMIA treats the non-target clients as the
    "OUT" population and fits a Gaussian N(mu_out, var_out). Before fitting,
    the paper removes unusually large reference values with the 3-sigma rule,
    because a non-target client can occasionally contain the same sample and
    would then no longer be a clean OUT reference.
    """
    sample_count = len(reference_measurements[0])
    mu_out: List[float] = []
    var_out: List[float] = []

    for sample_idx in range(sample_count):
        sample_values = [row[sample_idx] for row in reference_measurements]

        # Eq. (8)-(9): remove very high reference measurements before fitting
        # Q_out. High values are suspicious because larger M(I | x,y) means the
        # update is more aligned with the candidate sample gradient.
        raw_mean = _mean(sample_values)
        raw_std = _std(sample_values, raw_mean)
        cutoff = raw_mean + config.outlier_std_factor * raw_std
        filtered_values = [value for value in sample_values if value <= cutoff]
        if not filtered_values:
            filtered_values = [min(sample_values)]

        # Eq. (10): estimate the OUT Gaussian mean/variance. The variance floor
        # is a numerical guard for tiny client counts or near-identical updates.
        filtered_mean = _mean(filtered_values)
        filtered_var = max(_variance(filtered_values, filtered_mean), config.min_variance)
        mu_out.append(filtered_mean)
        var_out.append(filtered_var)
    return mu_out, var_out


def score_round(round_measurements: FedMIARoundMeasurements, config: Optional[FedMIAConfig] = None):
    if config is None:
        config = FedMIAConfig()
    mu_out, var_out = _estimate_null_distribution(round_measurements.reference_measurements, config)

    # Eq. (11): score the target update by the Gaussian CDF under Q_out.
    # A high CDF means the target update's measurement is unusually large
    # compared with non-target clients, which is member-like for this attack.
    round_scores = [
        normal_cdf(target_value, mu, var)
        for target_value, mu, var in zip(round_measurements.target_measurements, mu_out, var_out)
    ]
    return round_scores, mu_out, var_out


def score_rounds(
    measurement_rounds: Sequence[FedMIARoundMeasurements],
    config: Optional[FedMIAConfig] = None,
) -> FedMIAScores:
    if config is None:
        config = FedMIAConfig()
    if not measurement_rounds:
        raise ValueError("At least one round is required for FedMIA scoring.")

    per_round_scores: List[List[float]] = []
    per_round_mu_out: List[List[float]] = []
    per_round_var_out: List[List[float]] = []

    for round_measurements in measurement_rounds:
        round_scores, mu_out, var_out = score_round(round_measurements, config)
        per_round_scores.append(round_scores)
        per_round_mu_out.append(mu_out)
        per_round_var_out.append(var_out)

    sample_count = len(per_round_scores[0])
    aggregate_scores = []
    for sample_idx in range(sample_count):
        # Eq. (12): combine evidence across communication rounds by averaging
        # the per-round Lambda scores for the same candidate sample.
        aggregate_scores.append(
            _mean([round_scores[sample_idx] for round_scores in per_round_scores])
        )

    # Algorithm 1: threshold the aggregated score with delta.
    predictions = [1 if score > config.threshold else 0 for score in aggregate_scores]
    return FedMIAScores(
        per_round_scores=per_round_scores,
        aggregate_scores=aggregate_scores,
        predictions=predictions,
        per_round_mu_out=per_round_mu_out,
        per_round_var_out=per_round_var_out,
    )


def _roc_curve(member_scores: Sequence[float], nonmember_scores: Sequence[float]):
    labeled_scores = [(float(score), 1) for score in member_scores] + [
        (float(score), 0) for score in nonmember_scores
    ]
    labeled_scores.sort(key=lambda item: item[0], reverse=True)

    positives = max(len(member_scores), 1)
    negatives = max(len(nonmember_scores), 1)

    tps = 0
    fps = 0
    fpr_values = [0.0]
    tpr_values = [0.0]

    for score, label in labeled_scores:
        if label == 1:
            tps += 1
        else:
            fps += 1
        fpr_values.append(fps / negatives)
        tpr_values.append(tps / positives)

    if fpr_values[-1] != 1.0 or tpr_values[-1] != 1.0:
        fpr_values.append(1.0)
        tpr_values.append(1.0)
    return fpr_values, tpr_values


def _trapezoid_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    area = 0.0
    for idx in range(1, len(xs)):
        area += (xs[idx] - xs[idx - 1]) * (ys[idx] + ys[idx - 1]) / 2.0
    return area


def _tprs_at_thresholds(fprs: Sequence[float], tprs: Sequence[float]) -> Dict[str, float]:
    thresholds = [10, 1, 0.1, 0.02, 0.01, 0.001, 0.0001]
    summary: Dict[str, float] = {}
    for threshold in thresholds:
        best_tpr = 0.0
        for fpr, tpr in zip(fprs, tprs):
            if fpr < threshold:
                best_tpr = tpr
        summary[str(threshold)] = best_tpr
    return summary


def _log_auc(fprs: Sequence[float], tprs: Sequence[float]) -> float:
    log_fprs = []
    log_tprs = []
    for fpr, tpr in zip(fprs, tprs):
        safe_fpr = max(float(fpr), 1e-5)
        safe_tpr = max(float(tpr), 1e-5)
        log_fprs.append((math.log10(safe_fpr) + 5.0) / 5.0)
        log_tprs.append((math.log10(safe_tpr) + 5.0) / 5.0)
    return _trapezoid_auc(log_fprs, log_tprs)


def evaluate_membership(
    member_rounds: Sequence[FedMIARoundMeasurements],
    nonmember_rounds: Sequence[FedMIARoundMeasurements],
    config: Optional[FedMIAConfig] = None,
) -> FedMIAEvaluation:
    if config is None:
        config = FedMIAConfig()
    member_scores = score_rounds(member_rounds, config)
    nonmember_scores = score_rounds(nonmember_rounds, config)

    fprs, tprs = _roc_curve(member_scores.aggregate_scores, nonmember_scores.aggregate_scores)
    auc = _trapezoid_auc(fprs, tprs)
    log_auc = _log_auc(fprs, tprs)
    return FedMIAEvaluation(
        member_scores=member_scores,
        nonmember_scores=nonmember_scores,
        auc=auc,
        log_auc=log_auc,
        tprs=_tprs_at_thresholds(fprs, tprs),
    )
