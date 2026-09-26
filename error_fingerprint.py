"""Deterministic rare-token error fingerprints shared by a target and a candidate.

A token counts when it is not in an English lexicon, appears in the target, and
recurs in at least two independent works by the same candidate. Tokens are
classified so that a likely misspelling, a British spelling convention, and an
unrecognized technical term are never weighed alike, and each fingerprint
records how many other candidates also use it, because shared field vocabulary
is not a personal habit. Grammar errors that are spelled correctly are out of
scope; the model still has to find those.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any

try:
    from spellchecker import SpellChecker
except ImportError:  # Optional: without a lexicon the diagnostic reports itself unavailable.
    SpellChecker = None


MIN_CANDIDATE_WORKS = 2
MIN_TOKEN_LENGTH = 4
MAX_FINGERPRINTS_PER_CANDIDATE = 12
MAX_TARGET_TOKENS = 25
_CLASS_ORDER = {"misspelling_like": 0, "variant_spelling": 1, "unrecognized_term": 2}
# British-to-American rewrites; a token is a variant spelling when a rewrite is a known word.
_VARIANT_RULES = (
    (r"our", "or"),
    (r"is(e|ed|es|ing|ation|ations)$", r"iz\1"),
    (r"ys(e|ed|es|ing)$", r"yz\1"),
    (r"tre(s?)$", r"ter\1"),
    (r"ence$", "ense"),
    (r"ogue(s?)$", r"og\1"),
    (r"ll(ed|ing|er|ers)$", r"l\1"),
    (r"ae", "e"),
    (r"oe", "e"),
)
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@lru_cache(maxsize=1)
def _lexicon() -> Any:
    return SpellChecker(language="en", distance=1) if SpellChecker is not None else None


def _clean(text: str) -> str:
    """Undo PDF artifacts that would otherwise look like misspellings."""
    value = unicodedata.normalize("NFKC", text)  # expands ligatures such as "ﬁ"
    value = "".join(character for character in unicodedata.normalize("NFKD", value) if not unicodedata.combining(character))
    value = value.replace("’", "'")
    return re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", value)  # rejoin words hyphenated across lines


def _lowercase_tokens(text: str) -> Counter[str]:
    """Count tokens that occur in lowercase; always-capitalized words are names."""
    return Counter(
        token
        for token in _WORD.findall(_clean(text))
        if token.islower() and len(token) >= MIN_TOKEN_LENGTH and "'" not in token
    )


def _classify(token: str, lexicon: Any) -> tuple[str, str | None]:
    for pattern, replacement in _VARIANT_RULES:
        variant = re.sub(pattern, replacement, token)
        if variant != token and not lexicon.unknown([variant]):
            return "variant_spelling", variant
    corrections = lexicon.candidates(token) or set()
    corrections.discard(token)
    known = sorted(word for word in corrections if not lexicon.unknown([word]))
    if known:
        # Prefer the most frequent correction; ties break alphabetically for stability.
        return "misspelling_like", max(known, key=lambda word: (lexicon.word_usage_frequency(word), -known.index(word)))
    return "unrecognized_term", None


def build_error_fingerprint_diagnostics(
    target_text: str, corpora: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Find non-dictionary tokens the target shares with 2+ works by one candidate."""
    lexicon = _lexicon()
    if lexicon is None:
        return {"available": False, "reason": "pyspellchecker is not installed; the lexical fingerprint check was skipped."}
    target_tokens = _lowercase_tokens(target_text)
    target_unknown = lexicon.unknown(target_tokens)
    works: dict[str, list[Counter[str]]] = {
        candidate: [_lowercase_tokens(str(sample.get("text", ""))) for sample in samples if str(sample.get("text", "")).strip()]
        for candidate, samples in corpora.items()
    }
    works = {candidate: counts for candidate, counts in works.items() if counts}
    if not target_unknown or not works:
        return {
            "available": False,
            "reason": "The target has no non-dictionary lowercase token or no candidate work was available.",
        }
    classes = {token: _classify(token, lexicon) for token in target_unknown}
    users = {
        token: {candidate for candidate, counts in works.items() if any(token in work for work in counts)}
        for token in target_unknown
    }
    candidates: dict[str, Any] = {}
    for candidate, counts in works.items():
        fingerprints = []
        for token in target_unknown:
            work_count = sum(1 for work in counts if token in work)
            if work_count < MIN_CANDIDATE_WORKS:
                continue
            token_class, suggestion = classes[token]
            fingerprints.append(
                {
                    "token": token,
                    "class": token_class,
                    "suggested_form": suggestion,
                    "target_count": target_tokens[token],
                    "candidate_works": work_count,
                    "other_candidates_using_it": len(users[token] - {candidate}),
                }
            )
        fingerprints.sort(
            key=lambda item: (
                _CLASS_ORDER[item["class"]],
                item["other_candidates_using_it"],
                -item["candidate_works"],
                item["token"],
            )
        )
        candidates[candidate] = {
            "works_checked": len(counts),
            "exclusive_misspelling_like": sum(
                1 for item in fingerprints if item["class"] == "misspelling_like" and not item["other_candidates_using_it"]
            ),
            "exclusive_variant_spellings": sum(
                1 for item in fingerprints if item["class"] == "variant_spelling" and not item["other_candidates_using_it"]
            ),
            "shared_fingerprints": fingerprints[:MAX_FINGERPRINTS_PER_CANDIDATE],
        }
    ranked = sorted(
        candidates,
        key=lambda name: (candidates[name]["exclusive_misspelling_like"], candidates[name]["exclusive_variant_spellings"]),
        reverse=True,
    )
    leader = ranked[0] if candidates[ranked[0]]["exclusive_misspelling_like"] else None
    target_listing = sorted(
        target_unknown,
        key=lambda token: (_CLASS_ORDER[classes[token][0]], -target_tokens[token], token),
    )[:MAX_TARGET_TOKENS]
    return {
        "available": True,
        "lexicon": "pyspellchecker English frequency lexicon (US spelling)",
        "method": (
            f"Lowercase tokens of {MIN_TOKEN_LENGTH}+ letters missing from the lexicon, present in the target and in at "
            f"least {MIN_CANDIDATE_WORKS} independent works by one candidate. misspelling_like: a known word is one edit "
            "away. variant_spelling: a British form of a known US word (a convention, not an error). "
            "unrecognized_term: no nearby known word, usually technical vocabulary."
        ),
        "target_nonstandard_tokens": [
            {
                "token": token,
                "class": classes[token][0],
                "suggested_form": classes[token][1],
                "count": target_tokens[token],
            }
            for token in target_listing
        ],
        "candidates": candidates,
        "leader": leader,
        "caveat": (
            "Only exclusive misspelling-like tokens are candidate-specific. A token other candidates also use is shared "
            "vocabulary; unrecognized terms are usually field jargon; British spelling is a common convention. PDF "
            "extraction, journal copy-editing, and coauthored private files can create or erase tokens. Spelled-correctly "
            "grammar errors are not detected here."
        ),
    }


def error_fingerprint_prompt_section(diagnostics: dict[str, Any]) -> str:
    return json.dumps(diagnostics, ensure_ascii=False, indent=2)
