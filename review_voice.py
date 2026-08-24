"""Deterministic referee-report voice diagnostics.

This module measures rhetorical habits that are useful for separating reviewers
who work in the same field.  It compares only review-like private samples.  Public
research papers are deliberately excluded because genre transfer would otherwise
turn subject expertise into a false style match.
"""

from __future__ import annotations

import json
import math
import re
from statistics import median, pstdev
from typing import Any


_PATTERNS: dict[str, tuple[str, ...]] = {
    "first_person_singular": (r"\bI\b", r"\bmy\b", r"\bme\b"),
    "first_person_plural": (r"\bwe\b", r"\bour\b", r"\bus\b"),
    "direct_author_reference": (
        r"\bthe authors?\b",
        r"\bthe manuscript\b",
        r"\bthe paper\b",
        r"\bthe article\b",
    ),
    "hedge": (
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bseems?\b",
        r"\bappears?\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bin my (?:view|opinion|judg(?:e)?ment)\b",
    ),
    "positive_evaluation": (
        r"\binteresting\b",
        r"\bimportant\b",
        r"\bnovel\b",
        r"\bvaluable\b",
        r"\buseful\b",
        r"\bclear(?:ly)?\b",
        r"\bwell[- ]written\b",
        r"\bimprovement\b",
    ),
    "negative_evaluation": (
        r"\bconcern(?:s)?\b",
        r"\bproblem(?:s)?\b",
        r"\bissue(?:s)?\b",
        r"\bflaw(?:s)?\b",
        r"\bincorrect\b",
        r"\binsufficient\b",
        r"\bnot (?:clear|convincing|enough|new|original|correct|suitable)\b",
        r"\blacks?\b",
        r"\bweak(?:ness|nesses)?\b",
    ),
    "recommendation": (
        r"\brecommend(?:ation|ed|ing)?\b",
        r"\baccept(?:ance|ed)?\b",
        r"\breject(?:ion|ed)?\b",
        r"\bpublish(?:able|ed|ing)?\b",
        r"\bsuitable for\b",
        r"\bjournal\b",
        r"\bmajor revision\b",
        r"\bminor revision\b",
    ),
    "suggestion": (
        r"\bshould\b",
        r"\bmust\b",
        r"\bneed(?:s|ed)? to\b",
        r"\bI (?:suggest|ask|encourage)\b",
        r"\bit would be (?:useful|helpful|better)\b",
    ),
    "contrast": (
        r"\bhowever\b",
        r"\bnevertheless\b",
        r"\balthough\b",
        r"\bwhile\b",
        r"\bon the other hand\b",
        r"\bbut\b",
    ),
    "citation_request": (
        r"\breference(?:s)?\b",
        r"\bcite(?:d|s)?\b",
        r"\bcitation(?:s)?\b",
        r"\bliterature\b",
        r"\bprevious work\b",
    ),
}

_RATE_CAPS = {
    "first_person_singular": 3.0,
    "first_person_plural": 3.0,
    "direct_author_reference": 5.0,
    "hedge": 5.0,
    "positive_evaluation": 5.0,
    "negative_evaluation": 5.0,
    "recommendation": 4.0,
    "suggestion": 5.0,
    "contrast": 5.0,
    "citation_request": 4.0,
}

_FEATURE_WEIGHTS = {
    "first_person_singular": 1.5,
    "first_person_plural": 0.8,
    "direct_author_reference": 1.0,
    "hedge": 1.2,
    "positive_evaluation": 1.0,
    "negative_evaluation": 1.2,
    "recommendation": 1.5,
    "suggestion": 1.2,
    "contrast": 0.8,
    "citation_request": 0.6,
    "question_density": 1.2,
    "semicolon_density": 0.5,
    "colon_density": 0.5,
    "parenthesis_density": 0.4,
    "mean_sentence_length": 0.9,
    "sentence_length_variation": 0.6,
    "paragraph_rhythm": 0.7,
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _pattern_count(text: str, patterns: tuple[str, ...]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def extract_review_voice(text: str) -> dict[str, Any]:
    """Return normalized, inspectable report-voice features for one sample."""
    words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", text)
    sentences = _sentences(text)
    paragraphs = [part for part in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if part.strip()]
    word_count = len(words)
    sentence_count = max(1, len(sentences))
    per_hundred = max(1.0, word_count / 100.0)
    raw_rates = {
        name: _pattern_count(text, patterns) / per_hundred
        for name, patterns in _PATTERNS.items()
    }
    lengths = [len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", sentence)) for sentence in sentences]
    lengths = [value for value in lengths if value]
    mean_sentence = sum(lengths) / len(lengths) if lengths else 0.0
    sentence_variation = pstdev(lengths) if len(lengths) > 1 else 0.0
    paragraph_sentence_counts = [len(_sentences(paragraph)) for paragraph in paragraphs] or [sentence_count]
    vector = {
        name: min(1.0, raw_rates[name] / cap)
        for name, cap in _RATE_CAPS.items()
    }
    vector.update(
        {
            "question_density": min(1.0, text.count("?") / sentence_count / 0.35),
            "semicolon_density": min(1.0, text.count(";") / sentence_count / 0.35),
            "colon_density": min(1.0, text.count(":") / sentence_count / 0.45),
            "parenthesis_density": min(1.0, (text.count("(") + text.count(")")) / 2 / sentence_count / 0.5),
            "mean_sentence_length": min(1.0, mean_sentence / 45.0),
            "sentence_length_variation": min(1.0, sentence_variation / 25.0),
            "paragraph_rhythm": min(1.0, (sum(paragraph_sentence_counts) / len(paragraph_sentence_counts)) / 8.0),
        }
    )
    marker_presence = {
        name: raw_rates[name] > 0
        for name in (
            "first_person_singular",
            "direct_author_reference",
            "hedge",
            "positive_evaluation",
            "negative_evaluation",
            "recommendation",
            "suggestion",
        )
    }
    review_likelihood = (
        0.05 * (40 <= word_count <= 8_000)
        + 0.08 * marker_presence["first_person_singular"]
        + 0.14 * marker_presence["direct_author_reference"]
        + 0.07 * marker_presence["hedge"]
        + 0.10 * marker_presence["positive_evaluation"]
        + 0.12 * marker_presence["negative_evaluation"]
        + 0.22 * marker_presence["recommendation"]
        + 0.10 * marker_presence["suggestion"]
        + 0.12 * bool(re.search(r"\b(?:result|proof|theorem|method|argument|section)\b", text, re.I))
    )
    behavior_marker_count = sum(marker_presence.values())
    review_like = (
        word_count >= 40 and behavior_marker_count >= 2 and review_likelihood >= 0.24
    ) or (
        word_count >= 25 and behavior_marker_count >= 4 and review_likelihood >= 0.38
    )
    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "review_likelihood": round(min(1.0, review_likelihood), 4),
        "review_behavior_marker_count": behavior_marker_count,
        "review_like": review_like,
        "raw_rates_per_100_words": {name: round(value, 4) for name, value in raw_rates.items()},
        "vector": {name: round(value, 6) for name, value in vector.items()},
    }


def _centroid(samples: list[dict[str, Any]]) -> dict[str, float]:
    return {
        feature: float(median([float(sample["vector"].get(feature, 0.0)) for sample in samples]))
        for feature in _FEATURE_WEIGHTS
    }


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    weight_total = sum(_FEATURE_WEIGHTS.values())
    return sum(
        weight * abs(float(left.get(feature, 0.0)) - float(right.get(feature, 0.0)))
        for feature, weight in _FEATURE_WEIGHTS.items()
    ) / weight_total


def build_review_voice_diagnostics(
    target_text: str,
    reference_corpus: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare report rhetoric against known private report-like samples only."""
    target = extract_review_voice(target_text)
    if not target["review_like"]:
        return {
            "available": False,
            "reason": "The target did not contain enough referee-report voice markers for a stable comparison.",
            "target": target,
        }
    candidates: dict[str, Any] = {}
    for candidate, samples in reference_corpus.items():
        extracted = []
        labels = []
        for sample in samples:
            text = str(sample.get("text", "")).strip()
            if not text:
                continue
            features = extract_review_voice(text)
            if features["review_like"]:
                extracted.append(features)
                labels.append(str(sample.get("name", "Private review sample")))
        if not extracted:
            continue
        candidate_centroid = _centroid(extracted)
        distance = _distance(target["vector"], candidate_centroid)
        pairwise_distances = [
            _distance(left["vector"], right["vector"])
            for index, left in enumerate(extracted)
            for right in extracted[index + 1 :]
        ]
        within_candidate_dispersion = (
            float(median(pairwise_distances)) if pairwise_distances else None
        )
        candidates[candidate] = {
            "review_like_sample_count": len(extracted),
            "sample_labels": labels[:6],
            "distance": round(distance, 6),
            "similarity_index": round(max(0.0, 1.0 - distance), 6),
            "within_candidate_dispersion": (
                round(within_candidate_dispersion, 6)
                if within_candidate_dispersion is not None
                else None
            ),
        }
    if len(candidates) < 2:
        return {
            "available": False,
            "reason": "Review-like private samples were available for fewer than two candidates.",
            "target": target,
            "candidates": candidates,
        }
    ranked = sorted(candidates, key=lambda name: candidates[name]["distance"])
    leader = ranked[0]
    runner_up = ranked[1]
    separation = candidates[runner_up]["distance"] - candidates[leader]["distance"]
    leader_sample_count = int(candidates[leader]["review_like_sample_count"])
    leader_dispersion = candidates[leader].get("within_candidate_dispersion")
    reliability = (
        "moderate"
        if leader_sample_count >= 2
        and leader_dispersion is not None
        and float(leader_dispersion) <= 0.12
        else "low"
    )
    return {
        "available": True,
        "method": "weighted distance over stance, hedging, recommendation, address, punctuation, sentence-length, and paragraph-rhythm features",
        "target": target,
        "candidates": candidates,
        "metric_leader": leader,
        "runner_up": runner_up,
        "leader_separation": round(separation, 6),
        "reliability": reliability,
        "caveat": (
            "This is an uncalibrated genre-specific diagnostic. It excludes public research papers and cannot "
            "distinguish journal templates, editorial rewriting, or intentional tone changes on its own. "
            "A one-reference leader is always low reliability, regardless of its separation."
        ),
    }


def review_voice_prompt_section(diagnostics: dict[str, Any]) -> str:
    return json.dumps(diagnostics, ensure_ascii=False, indent=2)
