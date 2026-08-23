"""Small, deterministic stylometry diagnostics used as evidence—not a verdict."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


FUNCTION_WORDS = """
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves also although however hence thus
therefore moreover nevertheless indeed rather quite still already yet since may might shall perhaps overall
due via within without among whereas whether
""".split()


def _normalized(text: str) -> str:
    value = text.casefold().replace("\u00ad", "")
    value = re.sub(r"[^a-z'.,;:!?()\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _ngrams(text: str, width: int) -> Counter[str]:
    value = _normalized(text)
    return Counter(value[index : index + width] for index in range(max(0, len(value) - width + 1)))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(left[key] * right[key] for key in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _word_frequencies(text: str) -> Counter[str]:
    counts = Counter(re.findall(r"[a-z]+", text.casefold()))
    total = sum(counts.values()) or 1
    return Counter({word: count / total for word, count in counts.items()})


def _distributed_word_windows(text: str, width: int, maximum: int = 8) -> list[str]:
    words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[.,;:!?()]", text)
    if not words:
        return []
    if len(words) <= width:
        return [" ".join(words)]
    last_start = max(0, len(words) - width)
    starts = sorted({round(index * last_start / max(1, maximum - 1)) for index in range(maximum)})
    return [" ".join(words[start : start + width]) for start in starts]


def _length_matched_character_scores(
    target: str,
    texts: dict[str, list[str]],
    target_word_count: int,
) -> dict[str, dict[str, float | int]]:
    """Compare very short targets with equally sized public-corpus windows."""
    width = max(40, target_word_count)
    target_profile = _ngrams(target, 4)
    output: dict[str, dict[str, float | int]] = {}
    for candidate, papers in texts.items():
        scores = sorted(
            (
                _cosine(target_profile, _ngrams(window, 4))
                for paper in papers
                for window in _distributed_word_windows(paper, width)
            ),
            reverse=True,
        )
        if not scores:
            continue
        middle = len(scores) // 2
        median = scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2
        output[candidate] = {
            "window_count": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "median": round(median, 4),
            "upper_quartile": round(scores[max(0, len(scores) // 4 - 1)], 4),
        }
    return output


def _burrows_distances(
    target: str,
    texts: dict[str, list[str]],
    *,
    feature_count: int = 160,
    fixed_features: list[str] | None = None,
) -> dict[str, float]:
    samples = [(candidate, _word_frequencies(text)) for candidate, values in texts.items() for text in values]
    if not samples:
        return {}
    aggregate: Counter[str] = Counter()
    for _, frequencies in samples:
        aggregate.update(frequencies)
    features = fixed_features or [word for word, _ in aggregate.most_common(feature_count)]
    means = {word: sum(freq[word] for _, freq in samples) / len(samples) for word in features}
    deviations: dict[str, float] = {}
    for word in features:
        variance = sum((freq[word] - means[word]) ** 2 for _, freq in samples) / len(samples)
        deviations[word] = math.sqrt(variance) or 1.0
    target_frequencies = _word_frequencies(target)
    target_z = {word: (target_frequencies[word] - means[word]) / deviations[word] for word in features}
    distances: dict[str, float] = {}
    for candidate in texts:
        candidate_samples = [freq for label, freq in samples if label == candidate]
        if not candidate_samples:
            continue
        centroid = {
            word: sum((freq[word] - means[word]) / deviations[word] for freq in candidate_samples)
            / len(candidate_samples)
            for word in features
        }
        distances[candidate] = sum(abs(target_z[word] - centroid[word]) for word in features) / len(features)
    return distances


def build_stylometry_diagnostics(
    target_text: str, corpora: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Compare a target with candidate corpora using three intentionally separate views."""
    usable = {
        candidate: [sample for sample in samples if str(sample.get("text", "")).strip()]
        for candidate, samples in corpora.items()
    }
    usable = {candidate: samples for candidate, samples in usable.items() if samples}
    word_count = len(re.findall(r"[A-Za-z]+", target_text))
    if len(usable) < 2 or word_count < 40:
        return {
            "available": False,
            "target_word_count": word_count,
            "reason": "At least two populated candidate corpora and a 40-word target are required.",
        }
    texts = {candidate: [str(sample["text"]) for sample in samples] for candidate, samples in usable.items()}
    target_profiles = {width: _ngrams(target_text, width) for width in (3, 4, 5)}
    burrows = _burrows_distances(target_text, texts)
    function_delta = _burrows_distances(target_text, texts, fixed_features=FUNCTION_WORDS)
    length_matched = (
        _length_matched_character_scores(target_text, texts, word_count)
        if word_count < 200
        else {}
    )
    candidates: dict[str, Any] = {}
    for candidate, samples in usable.items():
        paper_scores = []
        for sample in samples:
            profiles = {width: _ngrams(str(sample["text"]), width) for width in (3, 4, 5)}
            similarity = sum(_cosine(target_profiles[width], profiles[width]) for width in profiles) / 3
            paper_scores.append(
                {
                    "title": str(sample.get("title") or sample.get("name") or "Untitled sample"),
                    "similarity": round(similarity, 4),
                }
            )
        paper_scores.sort(key=lambda item: item["similarity"], reverse=True)
        candidates[candidate] = {
            "sample_count": len(paper_scores),
            "character_ngram_mean": round(
                sum(item["similarity"] for item in paper_scores) / len(paper_scores), 4
            ),
            "character_ngram_best_three_mean": round(
                sum(item["similarity"] for item in paper_scores[:3]) / min(3, len(paper_scores)), 4
            ),
            "burrows_delta": round(burrows[candidate], 4),
            "function_word_delta": round(function_delta[candidate], 4),
            "closest_samples": paper_scores[:3],
        }
        if candidate in length_matched:
            candidates[candidate]["length_matched_character"] = length_matched[candidate]
    metric_specs = {
        "character_ngram_best_three_mean": True,
        "burrows_delta": False,
        "function_word_delta": False,
    }
    if length_matched:
        metric_specs["length_matched_character_median"] = True
        for candidate in candidates:
            candidates[candidate]["length_matched_character_median"] = candidates[candidate][
                "length_matched_character"
            ]["median"]
    leaders: dict[str, str] = {}
    for metric, higher_is_closer in metric_specs.items():
        ranked = sorted(
            candidates,
            key=lambda name: candidates[name][metric],
            reverse=higher_is_closer,
        )
        leaders[metric] = ranked[0]
        for rank, name in enumerate(ranked, start=1):
            candidates[name].setdefault("metric_ranks", {})[metric] = rank
    if word_count < 150:
        reliability = "very low"
    elif word_count < 350:
        reliability = "low"
    elif word_count < 1_000:
        reliability = "moderate"
    else:
        reliability = "moderate-to-good"
    return {
        "available": True,
        "target_word_count": word_count,
        "short_sample_reliability": reliability,
        "methods": {
            "character_ngram": "Cosine similarity over normalized character 3-, 4-, and 5-grams; higher is closer.",
            "burrows_delta": "Mean absolute standardized distance over the 160 most frequent corpus words; lower is closer.",
            "function_word_delta": "Burrows-style distance restricted to an English function-word list; lower is closer.",
            **(
                {
                    "length_matched_character": (
                        "For targets under 200 words only, character 4-gram cosine against distributed candidate "
                        "windows of the same word length; median is ranked higher-is-closer. This controls a major "
                        "short-sample length bias but is correlated with the ordinary character n-gram view and "
                        "must not be counted as an independent evidence family."
                    )
                }
                if length_matched
                else {}
            ),
        },
        "metric_leaders": leaders,
        "candidates": candidates,
        "caveat": (
            "These are uncalibrated diagnostics, not probabilities. Topic, genre, PDF extraction, equations, "
            "unequal corpus sizes, and short targets can dominate. Use agreement across methods as supporting "
            "evidence only; a single metric must never override stronger error, provenance, or reviewer-role evidence."
        ),
    }


def stylometry_prompt_section(diagnostics: dict[str, Any]) -> str:
    import json

    return json.dumps(diagnostics, ensure_ascii=False, indent=2)
