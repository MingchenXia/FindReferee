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


_KEPT_WORDS = frozenset(FUNCTION_WORDS)
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

# Character views that measure overlapping signal. Each family casts one vote;
# its first available metric speaks for it, so the topic-masked view overrides
# raw character n-grams, and legacy keys keep older stored reports readable.
VIEW_FAMILIES: dict[str, tuple[str, ...]] = {
    "character": (
        "topic_masked_character_best_three_mean",
        "character_ngram_best_three_mean",
        "length_matched_character_median",
    ),
    "most_frequent_words": ("cosine_delta", "burrows_delta"),
    "function_words": ("function_word_cosine_delta", "function_word_delta"),
}


def view_family_leaders(metric_leaders: dict[str, Any]) -> dict[str, str]:
    """Collapse correlated metric leaders into one leader per view family."""
    leaders: dict[str, str] = {}
    for family, metrics in VIEW_FAMILIES.items():
        for metric in metrics:
            leader = str(metric_leaders.get(metric) or "").strip()
            if leader:
                leaders[family] = leader
                break
    return leaders


def _normalized(text: str) -> str:
    value = text.casefold().replace("\u00ad", "")
    value = re.sub(r"[^a-z'.,;:!?()\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _topic_masked(text: str) -> str:
    """Text distortion (Stamatatos 2017, DV-MA) with subject vocabulary removed.

    Function words stay; every other word becomes asterisks of the same length.
    Punctuation, word-length rhythm, and function-word order survive, while the
    technical terms that make same-field candidates look alike do not.
    """
    value = text.casefold().replace("\u00ad", "").replace("’", "'")
    value = _WORD.sub(lambda match: match.group(0) if match.group(0) in _KEPT_WORDS else "*" * len(match.group(0)), value)
    value = re.sub(r"[^a-z*'.,;:!?()\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_NGRAM_WIDTHS = (3, 4, 5)
# A character n-gram profile paired with its Euclidean norm, so the norm of a
# profile that is compared many times (the target) is computed only once.
_Profile = tuple[Counter[str], float]


def _ngram_counts(normalized: str, width: int) -> Counter[str]:
    return Counter(normalized[index : index + width] for index in range(max(0, len(normalized) - width + 1)))


def _ngrams(text: str, width: int) -> Counter[str]:
    return _ngram_counts(_normalized(text), width)


def _profile(counts: Counter[str]) -> _Profile:
    return counts, math.sqrt(sum(value * value for value in counts.values()))


def _masked_ngram_counts(masked: str, width: int) -> Counter[str]:
    # Grams made only of mask characters record nothing but "long words here",
    # yet they would dominate the counts and push every similarity toward 1.
    counts = _ngram_counts(masked, width)
    for gram in [gram for gram in counts if not gram.strip("* ")]:
        del counts[gram]
    return counts


def _ngram_profiles(text: str, *, topic_masked: bool = False) -> dict[int, _Profile]:
    """Normalize a text once and build every character n-gram view from it."""
    if topic_masked:
        masked = _topic_masked(text)
        return {width: _profile(_masked_ngram_counts(masked, width)) for width in _NGRAM_WIDTHS}
    normalized = _normalized(text)
    return {width: _profile(_ngram_counts(normalized, width)) for width in _NGRAM_WIDTHS}


def _mean_similarity(left: dict[int, _Profile], right: dict[int, _Profile]) -> float:
    return sum(_profile_cosine(left[width], right[width]) for width in _NGRAM_WIDTHS) / len(_NGRAM_WIDTHS)


def _profile_cosine(left: _Profile, right: _Profile) -> float:
    left_counts, left_norm = left
    right_counts, right_norm = right
    if not left_norm or not right_norm:
        return 0.0
    # Integer dot product: iterating the smaller profile gives the exact same value.
    small, large = (left_counts, right_counts) if len(left_counts) <= len(right_counts) else (right_counts, left_counts)
    numerator = sum(value * large.get(key, 0) for key, value in small.items())
    return numerator / (left_norm * right_norm)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    return _profile_cosine(_profile(left), _profile(right))


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
    target_profile: _Profile | None = None,
) -> dict[str, dict[str, float | int]]:
    """Compare very short targets with equally sized public-corpus windows."""
    width = max(40, target_word_count)
    target_profile = target_profile or _profile(_ngrams(target, 4))
    output: dict[str, dict[str, float | int]] = {}
    for candidate, papers in texts.items():
        scores = sorted(
            (
                _profile_cosine(target_profile, _profile(_ngrams(window, 4)))
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


def _most_frequent_words(samples: list[tuple[str, Counter[str]]], feature_count: int) -> list[str]:
    aggregate: Counter[str] = Counter()
    for _, frequencies in samples:
        aggregate.update(frequencies)
    return [word for word, _ in aggregate.most_common(feature_count)]


def _cosine_distance(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 1.0 - numerator / norms if norms else 1.0


def _cosine_delta_from_frequencies(
    target_frequencies: Counter[str],
    samples: list[tuple[str, Counter[str]]],
    features: list[str],
) -> dict[str, float]:
    """Cosine Delta (Smith & Aldridge 2011) over precomputed relative word frequencies.

    Frequencies are z-scored against the pooled comparison works, as in Burrows's
    Delta, but the target and each candidate centroid are compared by the angle
    between their z-profiles instead of the mean absolute difference. The angle
    ignores how extreme a profile is overall and has been more robust across
    corpora and feature counts (Evert et al. 2017). 0 is identical, lower is closer.
    """
    if not samples:
        return {}
    means = [sum(freq[word] for _, freq in samples) / len(samples) for word in features]
    deviations = [
        math.sqrt(sum((freq[word] - mean) ** 2 for _, freq in samples) / len(samples)) or 1.0
        for word, mean in zip(features, means)
    ]

    def z_profile(frequencies: Counter[str]) -> list[float]:
        return [(frequencies[word] - mean) / deviation for word, mean, deviation in zip(features, means, deviations)]

    target_z = z_profile(target_frequencies)
    grouped: dict[str, list[list[float]]] = {}
    for label, frequencies in samples:
        grouped.setdefault(label, []).append(z_profile(frequencies))
    return {
        candidate: _cosine_distance(target_z, [sum(values) / len(profiles) for values in zip(*profiles)])
        for candidate, profiles in grouped.items()
    }


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
    # Each text is normalized, n-grammed, and word-counted exactly once; both
    # Delta views share the same relative word frequencies.
    target_profiles = _ngram_profiles(target_text)
    target_masked_profiles = _ngram_profiles(target_text, topic_masked=True)
    target_frequencies = _word_frequencies(target_text)
    sample_frequencies = [
        (candidate, _word_frequencies(text)) for candidate, values in texts.items() for text in values
    ]
    cosine_delta = _cosine_delta_from_frequencies(
        target_frequencies, sample_frequencies, _most_frequent_words(sample_frequencies, 160)
    )
    function_delta = _cosine_delta_from_frequencies(target_frequencies, sample_frequencies, FUNCTION_WORDS)
    length_matched = (
        _length_matched_character_scores(target_text, texts, word_count, target_profiles[4])
        if word_count < 200
        else {}
    )
    candidates: dict[str, Any] = {}
    for candidate, samples in usable.items():
        paper_scores = []
        masked_scores = []
        for sample in samples:
            text = str(sample["text"])
            similarity = _mean_similarity(target_profiles, _ngram_profiles(text))
            masked_scores.append(round(_mean_similarity(target_masked_profiles, _ngram_profiles(text, topic_masked=True)), 4))
            paper_scores.append(
                {
                    "title": str(sample.get("title") or sample.get("name") or "Untitled sample"),
                    "similarity": round(similarity, 4),
                }
            )
        paper_scores.sort(key=lambda item: item["similarity"], reverse=True)
        masked_scores.sort(reverse=True)
        candidates[candidate] = {
            "sample_count": len(paper_scores),
            "character_ngram_mean": round(
                sum(item["similarity"] for item in paper_scores) / len(paper_scores), 4
            ),
            "character_ngram_best_three_mean": round(
                sum(item["similarity"] for item in paper_scores[:3]) / min(3, len(paper_scores)), 4
            ),
            "topic_masked_character_mean": round(sum(masked_scores) / len(masked_scores), 4),
            "topic_masked_character_best_three_mean": round(
                sum(masked_scores[:3]) / min(3, len(masked_scores)), 4
            ),
            "cosine_delta": round(cosine_delta[candidate], 4),
            "function_word_cosine_delta": round(function_delta[candidate], 4),
            "closest_samples": paper_scores[:3],
        }
        if candidate in length_matched:
            candidates[candidate]["length_matched_character"] = length_matched[candidate]
    metric_specs = {
        "character_ngram_best_three_mean": True,
        "topic_masked_character_best_three_mean": True,
        "cosine_delta": False,
        "function_word_cosine_delta": False,
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
    topic_ablation = {
        "raw_character_leader": leaders["character_ngram_best_three_mean"],
        "topic_masked_character_leader": leaders["topic_masked_character_best_three_mean"],
        "agrees": leaders["character_ngram_best_three_mean"] == leaders["topic_masked_character_best_three_mean"],
    }
    topic_ablation["note"] = (
        "The character-level leader survives removal of subject vocabulary."
        if topic_ablation["agrees"]
        else "Masking subject vocabulary changes the character-level leader, so raw character similarity is "
        "likely topic-driven; the topic-masked view speaks for the character family."
    )
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
            "topic_masked_character": (
                "Text distortion (Stamatatos 2017): every word outside an English function-word list is replaced by "
                "asterisks of the same length before the same character n-gram cosine; higher is closer. It keeps "
                "punctuation, function-word order, and word-length rhythm while removing subject vocabulary, so it is "
                "a deterministic expertise-ablation check. It belongs to the same character family as the raw view."
            ),
            "cosine_delta": (
                "Cosine Delta (Smith & Aldridge 2011): cosine distance between z-scored frequency profiles of the "
                "160 most frequent corpus words, target versus each candidate centroid; 0 is identical, lower is closer."
            ),
            "function_word_cosine_delta": "Cosine Delta restricted to an English function-word list; lower is closer.",
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
        "view_family_leaders": view_family_leaders(leaders),
        "topic_ablation": topic_ablation,
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
