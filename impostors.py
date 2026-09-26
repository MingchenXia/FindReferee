"""General Impostors verification (Koppel & Winter 2014) for an open candidate set.

Every iteration keeps a random half of the character 4-gram features and asks
which author's closest document is most similar to the target. Listed
candidates compete with same-field "impostor" authors who are not on the list,
so the share of iterations won by an impostor is a reproducible reference for
the no-listed-candidate alternative. It is an uncalibrated diagnostic, not a
probability.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from operator import itemgetter
from typing import Any

from stylometry import _ngram_counts, _normalized


ITERATIONS = 100
FEATURE_COUNT = 600
FEATURE_FRACTION = 0.5
NGRAM_WIDTH = 4
# A fixed seed makes the diagnostic reproducible for identical inputs.
RANDOM_SEED = 2014


def _texts(samples: list[dict[str, Any]]) -> list[str]:
    return [str(sample.get("text", "")) for sample in samples if str(sample.get("text", "")).strip()]


def _relative_frequencies(counts: Counter[str], features: list[str]) -> list[float]:
    total = sum(counts.values()) or 1
    return [counts.get(feature, 0) / total for feature in features]


def build_impostor_diagnostics(
    target_text: str,
    candidate_corpora: dict[str, list[dict[str, Any]]],
    impostor_corpora: dict[str, list[dict[str, Any]]] | None = None,
    *,
    iterations: int = ITERATIONS,
    feature_count: int = FEATURE_COUNT,
    feature_fraction: float = FEATURE_FRACTION,
) -> dict[str, Any]:
    """Compare listed candidates with outside same-field authors over random feature subsets."""
    groups: list[tuple[str, bool, list[str]]] = [
        (name, True, texts) for name, samples in candidate_corpora.items() if (texts := _texts(samples))
    ]
    groups += [
        (name, False, texts)
        for name, samples in (impostor_corpora or {}).items()
        if name not in candidate_corpora and (texts := _texts(samples))
    ]
    candidate_indexes = [index for index, (_, is_candidate, _) in enumerate(groups) if is_candidate]
    impostor_indexes = [index for index, (_, is_candidate, _) in enumerate(groups) if not is_candidate]
    word_count = len(re.findall(r"[A-Za-z]+", target_text))
    if word_count < 40 or not candidate_indexes or len(groups) < 2:
        return {
            "available": False,
            "reason": "A 40-word target and public solo works from at least two authors, one of them listed, are required.",
        }

    target_counts = _ngram_counts(_normalized(target_text), NGRAM_WIDTH)
    document_counts = [
        (index, _ngram_counts(_normalized(text), NGRAM_WIDTH))
        for index, (_, _, texts) in enumerate(groups)
        for text in texts
    ]
    pooled: Counter[str] = Counter(target_counts)
    for _, counts in document_counts:
        pooled.update(counts)
    # Sort explicitly so the feature order, and therefore every sampled subset,
    # is independent of hash randomization.
    features = [feature for feature, _ in sorted(pooled.items(), key=lambda item: (-item[1], item[0]))[:feature_count]]
    subset_size = int(len(features) * feature_fraction)
    if subset_size < 2:
        return {"available": False, "reason": "Too few shared character features for a stable comparison."}

    target_vector = _relative_frequencies(target_counts, features)
    documents = []
    for index, counts in document_counts:
        vector = _relative_frequencies(counts, features)
        documents.append((index, vector, [abs(left - right) for left, right in zip(target_vector, vector)]))

    rng = random.Random(RANDOM_SEED)
    feature_indexes = range(len(features))
    wins = [0.0] * len(groups)
    beats_every_impostor = [0] * len(groups)
    for _ in range(iterations):
        pick = itemgetter(*rng.sample(feature_indexes, subset_size))
        target_mass = sum(pick(target_vector))
        best = [-1.0] * len(groups)
        for index, vector, difference in documents:
            # Min-max similarity: sum(min)/sum(max) = (a + b - |a-b|) / (a + b + |a-b|).
            mass = target_mass + sum(pick(vector))
            distance = sum(pick(difference))
            similarity = (mass - distance) / (mass + distance) if mass + distance else 0.0
            if similarity > best[index]:
                best[index] = similarity
        top = max(best)
        winners = [index for index, value in enumerate(best) if value == top]
        for index in winners:
            wins[index] += 1 / len(winners)
        if impostor_indexes:
            strongest_impostor = max(best[index] for index in impostor_indexes)
            for index in candidate_indexes:
                if best[index] > strongest_impostor:
                    beats_every_impostor[index] += 1

    candidates = {
        groups[index][0]: {
            "documents": len(groups[index][2]),
            "attribution_share": round(wins[index] / iterations, 4),
            "verification_score": (
                round(beats_every_impostor[index] / iterations, 4) if impostor_indexes else None
            ),
        }
        for index in candidate_indexes
    }
    leader = max(candidates, key=lambda name: candidates[name]["attribution_share"])
    external_win_rate = (
        round(sum(wins[index] for index in impostor_indexes) / iterations, 4) if impostor_indexes else None
    )
    if not impostor_indexes:
        reliability = "closed set only"
    elif word_count < 350 or len(impostor_indexes) < 3:
        reliability = "low"
    else:
        reliability = "moderate"
    return {
        "available": True,
        "method": (
            f"General Impostors (Koppel & Winter 2014): {iterations} iterations, each keeping a random "
            f"{feature_fraction:.0%} of the {len(features)} most frequent character {NGRAM_WIDTH}-grams and "
            "min-max similarity to each author's closest public solo work."
        ),
        "target_word_count": word_count,
        "external_impostor_authors": [groups[index][0] for index in impostor_indexes],
        "external_impostor_documents": sum(len(groups[index][2]) for index in impostor_indexes),
        "candidates": candidates,
        "leader": leader,
        "external_impostor_win_rate": external_win_rate,
        "reliability": reliability,
        "interpretation": (
            "attribution_share: how often the candidate is the closest author of all. verification_score: how "
            "often it beats every outside same-field impostor. external_impostor_win_rate: how often an unlisted "
            "author is closest, a reference for the no-listed-candidate alternative."
            if impostor_indexes
            else "No outside impostor corpus was available, so this only measures how stable the closed-set "
            "ranking is under feature resampling; it says nothing about unlisted authors."
        ),
        "caveat": (
            "Uncalibrated and correlated with the character n-gram family. The comparison works are research "
            "papers, so a referee-report target is cross-genre. Treat it as one weak reference, never as proof."
        ),
    }


def impostor_prompt_section(diagnostics: dict[str, Any]) -> str:
    return json.dumps(diagnostics, ensure_ascii=False, indent=2)
