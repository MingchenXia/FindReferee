"""Transparent metrics for blind authorship-attribution regression tests."""

from __future__ import annotations

import math
from typing import Any, Iterable


EPSILON = 1e-9
NO_LISTED_CANDIDATE = "No listed candidate"


def probability_distribution(result: dict[str, Any]) -> dict[str, float]:
    distribution = {
        str(item.get("candidate", "")): max(0.0, float(item.get("probability", 0) or 0))
        for item in result.get("candidate_evaluations", [])
        if isinstance(item, dict) and str(item.get("candidate", "")).strip()
    }
    distribution[NO_LISTED_CANDIDATE] = max(
        0.0, float(result.get("no_listed_candidate_probability", 0) or 0)
    )
    total = sum(distribution.values())
    if total <= 0:
        return distribution
    return {name: value / total for name, value in distribution.items()}


def score_case(result: dict[str, Any], expected_author: str) -> dict[str, Any]:
    """Score one blind result after the expected label has been withheld from inference."""
    distribution = probability_distribution(result)
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    expected_probability = distribution.get(expected_author, 0.0)
    # Use competition ranking so an expected label tied for the highest score has
    # rank 1, but do not silently turn insertion order into a unique Top-1 win.
    expected_rank = (
        1
        + sum(
            probability > expected_probability + EPSILON
            for name, probability in distribution.items()
            if name != expected_author
        )
        if expected_author in distribution
        else len(ranked) + 1
    )
    strongest_alternative = max(
        (probability for name, probability in distribution.items() if name != expected_author),
        default=0.0,
    )
    brier = sum(
        (probability - (1.0 if name == expected_author else 0.0)) ** 2
        for name, probability in distribution.items()
    )
    entropy = -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0
    )
    maximum_probability = max(distribution.values(), default=0.0)
    top1_including_ties = bool(
        expected_author in distribution
        and expected_probability >= maximum_probability - EPSILON
    )
    top1_tie = bool(
        top1_including_ties
        and sum(
            abs(probability - maximum_probability) <= EPSILON
            for probability in distribution.values()
        )
        > 1
    )
    top1_correct = bool(top1_including_ties and not top1_tie)
    determination = result.get("determination") if isinstance(result.get("determination"), dict) else {}
    determination_status = str(determination.get("status", "")).strip()
    numerical_separation = bool(
        ranked
        and expected_probability >= 0.55
        and expected_probability - strongest_alternative >= 0.20
    )
    precise_claim = (
        determination_status == "meaningfully_separated"
        if determination_status
        else numerical_separation
    )
    return {
        "top1_correct": top1_correct,
        "top1_including_ties": top1_including_ties,
        "top1_tie": top1_tie,
        "expected_rank": expected_rank,
        "reciprocal_rank": 1.0 / expected_rank,
        "expected_probability": expected_probability,
        "true_class_margin": expected_probability - strongest_alternative,
        "log_loss": -math.log(max(EPSILON, expected_probability)),
        "brier_score": brier,
        "entropy_nats": entropy,
        "determination_status": determination_status or "legacy_threshold_fallback",
        "precise_claim": precise_claim,
        "precise_claim_correct": bool(precise_claim and top1_correct),
        "precise_claim_wrong": bool(precise_claim and not top1_correct),
        "non_precise": not precise_claim,
        "unable_to_determine": determination_status == "unable_to_determine",
        "decisive_correct": bool(precise_claim and top1_correct),
    }


def jensen_shannon_divergence(
    left: dict[str, float], right: dict[str, float]
) -> float:
    labels = set(left) | set(right)
    left_total = sum(max(0.0, left.get(label, 0.0)) for label in labels) or 1.0
    right_total = sum(max(0.0, right.get(label, 0.0)) for label in labels) or 1.0
    left_normalized = {label: max(0.0, left.get(label, 0.0)) / left_total for label in labels}
    right_normalized = {label: max(0.0, right.get(label, 0.0)) / right_total for label in labels}
    midpoint = {
        label: (left_normalized[label] + right_normalized[label]) / 2.0 for label in labels
    }

    def divergence(source: dict[str, float]) -> float:
        return sum(
            probability * math.log(probability / midpoint[label])
            for label, probability in source.items()
            if probability > 0 and midpoint[label] > 0
        )

    return (divergence(left_normalized) + divergence(right_normalized)) / 2.0


def aggregate_scores(scores: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    rows = list(scores)
    if not rows:
        return {"case_count": 0}

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    precise_claim_count = sum(bool(row.get("precise_claim")) for row in rows)
    precise_claim_correct = sum(bool(row.get("precise_claim_correct")) for row in rows)
    precise_claim_wrong = sum(bool(row.get("precise_claim_wrong")) for row in rows)
    return {
        "case_count": len(rows),
        "top1_accuracy": mean("top1_correct"),
        "top1_including_ties_accuracy": mean("top1_including_ties"),
        "top1_tie_rate": mean("top1_tie"),
        "decisive_accuracy": mean("decisive_correct"),
        "precise_claim_count": precise_claim_count,
        "precise_claim_coverage": precise_claim_count / len(rows),
        "precise_claim_precision": (
            precise_claim_correct / precise_claim_count if precise_claim_count else 0.0
        ),
        "false_precise_claim_rate": precise_claim_wrong / len(rows),
        "non_precise_rate": sum(bool(row.get("non_precise")) for row in rows) / len(rows),
        "unable_to_determine_rate": (
            sum(bool(row.get("unable_to_determine")) for row in rows) / len(rows)
        ),
        "mean_reciprocal_rank": mean("reciprocal_rank"),
        "mean_expected_probability": mean("expected_probability"),
        "mean_true_class_margin": mean("true_class_margin"),
        "mean_log_loss": mean("log_loss"),
        "mean_brier_score": mean("brier_score"),
        "mean_entropy_nats": mean("entropy_nats"),
    }


MIN_TEMPERATURE = 0.25
MAX_TEMPERATURE = 4.0


def temperature_scaled(distribution: dict[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scale a probability distribution: T > 1 flattens it, T < 1 sharpens it."""
    powered = {
        label: probability ** (1.0 / temperature) if probability > 0 else 0.0
        for label, probability in distribution.items()
    }
    total = sum(powered.values())
    return {label: value / total for label, value in powered.items()} if total > 0 else dict(distribution)


def uncalibrated_distribution(result: dict[str, Any]) -> dict[str, float]:
    """The distribution before any stored calibration, so refitting never compounds it."""
    calibration = result.get("probability_calibration")
    if isinstance(calibration, dict) and calibration.get("applied"):
        stored = calibration.get("uncalibrated_distribution")
        if isinstance(stored, dict) and stored:
            return {str(label): max(0.0, float(value)) for label, value in stored.items()}
    return probability_distribution(result)


def fit_temperature(cases: Iterable[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    """Fit one temperature that minimizes mean log loss of the expected labels.

    A golden-section search over log T is enough because the objective is a
    smooth one-parameter curve on the bounded range.
    """
    rows = [(uncalibrated_distribution(result), expected) for result, expected in cases]
    if not rows:
        raise ValueError("At least one scored case is required to fit a temperature.")

    def mean_log_loss(temperature: float) -> float:
        return sum(
            -math.log(max(EPSILON, temperature_scaled(distribution, temperature).get(expected, 0.0)))
            for distribution, expected in rows
        ) / len(rows)

    low, high = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_loss, right_loss = mean_log_loss(math.exp(left)), mean_log_loss(math.exp(right))
    for _ in range(60):
        if left_loss <= right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = mean_log_loss(math.exp(left))
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = mean_log_loss(math.exp(right))
    temperature = math.exp((low + high) / 2.0)
    return {
        "method": "temperature",
        "temperature": round(temperature, 4),
        "case_count": len(rows),
        "mean_log_loss_before": round(mean_log_loss(1.0), 6),
        "mean_log_loss_after": round(mean_log_loss(temperature), 6),
    }


def repeated_run_stability(results: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    distributions = [probability_distribution(result) for result in results]
    pairwise = [
        jensen_shannon_divergence(distributions[left], distributions[right])
        for left in range(len(distributions))
        for right in range(left + 1, len(distributions))
    ]
    return {
        "run_count": len(distributions),
        "pair_count": len(pairwise),
        "mean_pairwise_js_divergence": sum(pairwise) / len(pairwise) if pairwise else 0.0,
        "max_pairwise_js_divergence": max(pairwise, default=0.0),
    }
