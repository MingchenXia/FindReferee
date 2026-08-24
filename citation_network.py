"""Bounded, time-aware citation-network priors for referee candidate discovery.

The graph is a reviewer-selection prior, never an authorship score.  It deliberately
keeps direct citations, second-order citations, and independence conflicts separate
so a model cannot silently turn shared expertise into identity evidence.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


API_ROOT = "https://api.semanticscholar.org/graph/v1"
API_HOST = "api.semanticscholar.org"
USER_AGENT = "FindReferee/1.1 (local research tool; citation-network attribution prior)"
CACHE_SECONDS = 30 * 24 * 60 * 60
MAX_DIRECT_REFERENCES = 100
MAX_SECOND_ORDER_SEEDS = 8
MAX_SECOND_ORDER_REFERENCES_PER_SEED = 60
MAX_CANDIDATE_RELATIONSHIP_LOOKUPS = 10
DIRECT_WEIGHT = 1.0
SECOND_ORDER_WEIGHT = 0.28
PREPUBLICATION_COAUTHOR_MULTIPLIER = 0.22
RATE_LIMIT_RETRIES = 3


def _cache_root() -> Path:
    configured = os.getenv("AUTHOR_ATTRIBUTION_CITATION_CACHE", "").strip()
    choices = [
        Path(configured).expanduser() if configured else None,
        Path.home() / ".cache" / "author-attribution" / "citation-network",
        Path(tempfile.gettempdir()) / "author-attribution-citation-network",
    ]
    for choice in choices:
        if choice is None:
            continue
        try:
            choice.mkdir(parents=True, exist_ok=True)
            return choice
        except OSError:
            continue
    raise OSError("No writable citation-network cache directory is available.")


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(character for character in normalized.casefold() if character.isalnum() or character.isspace())


def _name_parts(value: str) -> list[str]:
    return [part for part in re.split(r"\s+", _fold(value).strip()) if part]


def _identity_key(value: str) -> str:
    """Create a cautious citation-graph key that joins full/initial first names.

    Initial-plus-surname identity is not proof.  Every merged key is explicitly
    marked as needing source verification in the prompt packet.
    """
    parts = _name_parts(value)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]}:{parts[0][0]}"


def _aliases(label: str) -> list[str]:
    aliases = [part.strip() for part in label.split("/") if part.strip()]
    return aliases or [label.strip()]


def _label_keys(label: str) -> set[str]:
    return {key for key in (_identity_key(alias) for alias in _aliases(label)) if key}


def _prefer_display_name(current: str, proposed: str) -> str:
    if not current:
        return proposed
    current_letters = sum(character.isalpha() for character in current)
    proposed_letters = sum(character.isalpha() for character in proposed)
    current_initials = len(re.findall(r"\b[A-Za-z]\.?\b", current))
    proposed_initials = len(re.findall(r"\b[A-Za-z]\.?\b", proposed))
    if proposed_initials < current_initials or (
        proposed_initials == current_initials and proposed_letters > current_letters
    ):
        return proposed
    return current


def _cache_path(url: str) -> Path:
    import hashlib

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return _cache_root() / f"{digest}.json"


def _request_json(url: str, *, timeout: int = 25) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() != API_HOST:
        raise ValueError("Only the official Semantic Scholar HTTPS API is allowed.")
    cache_path = _cache_path(url)
    if cache_path.is_file() and time.time() - cache_path.stat().st_mtime < CACHE_SECONDS:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt + 1 < RATE_LIMIT_RETRIES:
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                try:
                    delay = max(1.0, min(10.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = min(8.0, 2.0 ** (attempt + 1))
                time.sleep(delay)
                continue
            if cache_path.is_file():
                return json.loads(cache_path.read_text(encoding="utf-8"))
            raise
        except Exception:
            if cache_path.is_file():
                return json.loads(cache_path.read_text(encoding="utf-8"))
            raise
    else:  # pragma: no cover - the loop either returns, raises, or breaks.
        raise RuntimeError("Semantic Scholar did not return citation metadata.")
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _api_url(path: str, parameters: dict[str, Any] | None = None) -> str:
    suffix = ""
    if parameters:
        suffix = "?" + urllib.parse.urlencode(parameters, doseq=True)
    return f"{API_ROOT}/{path.lstrip('/')}{suffix}"


def _extract_identifier(source: str) -> str | None:
    arxiv_patterns = (
        r"(?i)arxiv\s*:\s*(\d{4}\.\d{4,5})(?:v\d+)?",
        r"(?i)arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?",
    )
    for pattern in arxiv_patterns:
        match = re.search(pattern, source)
        if match:
            return f"ARXIV:{match.group(1)}"
    doi_match = re.search(r"(?i)\b(?:doi\s*:\s*|https?://doi\.org/)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", source)
    if doi_match:
        return f"DOI:{doi_match.group(1).rstrip('.,;)')}"
    return None


def _metadata_title(document: dict[str, Any] | None) -> str:
    if not document:
        return ""
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    for key, value in metadata.items():
        if str(key).strip("/").casefold() == "title" and str(value).strip():
            return re.sub(r"\s+", " ", str(value)).strip()
    return ""


def _resolve_subject(document: dict[str, Any] | None, context_note: str) -> dict[str, Any]:
    if not document and not context_note.strip():
        raise ValueError("No underlying manuscript or identifying context was supplied.")
    metadata = document.get("metadata", {}) if document else {}
    opening = str(document.get("text", ""))[:20_000] if document else ""
    source = f"{context_note}\n{json.dumps(metadata, ensure_ascii=False)}\n{opening}"
    identifier = _extract_identifier(source)
    fields = "title,year,authors,referenceCount"
    if identifier:
        return _request_json(_api_url(f"paper/{urllib.parse.quote(identifier, safe=':')}", {"fields": fields}))
    title = _metadata_title(document)
    if not title:
        raise ValueError("No arXiv ID, DOI, or PDF title was available to resolve the underlying manuscript.")
    payload = _request_json(
        _api_url("paper/search", {"query": title, "limit": 5, "fields": fields})
    )
    results = payload.get("data") if isinstance(payload.get("data"), list) else []
    if not results:
        raise ValueError("The underlying manuscript could not be resolved in Semantic Scholar.")
    best = max(
        results,
        key=lambda item: SequenceMatcher(None, _fold(title), _fold(str(item.get("title", "")))).ratio(),
    )
    similarity = SequenceMatcher(None, _fold(title), _fold(str(best.get("title", "")))).ratio()
    if similarity < 0.82:
        raise ValueError("The underlying manuscript title did not resolve unambiguously.")
    return best


def _reference_rows(paper_id: str, limit: int) -> list[dict[str, Any]]:
    payload = _request_json(
        _api_url(
            f"paper/{paper_id}/references",
            {
                "limit": limit,
                "fields": "title,year,authors,isInfluential",
            },
        )
    )
    return payload.get("data") if isinstance(payload.get("data"), list) else []


def _year_allowed(year: Any, subject_year: int | None) -> bool:
    if subject_year is None or year in (None, ""):
        return True
    try:
        return int(year) <= subject_year
    except (TypeError, ValueError):
        return True


def _paper_authors(paper: dict[str, Any]) -> list[dict[str, Any]]:
    authors = paper.get("authors")
    return [author for author in authors if isinstance(author, dict)] if isinstance(authors, list) else []


def _add_author_evidence(
    ledger: dict[str, dict[str, Any]],
    paper: dict[str, Any],
    *,
    tier: str,
    influential: bool,
    seed_title: str = "",
) -> None:
    authors = _paper_authors(paper)
    if not authors:
        return
    paper_weight = (1.15 if influential else 1.0) / math.sqrt(len(authors))
    title = re.sub(r"\s+", " ", str(paper.get("title", ""))).strip()
    for author in authors:
        name = re.sub(r"\s+", " ", str(author.get("name", ""))).strip()
        key = _identity_key(name)
        if not key:
            continue
        item = ledger.setdefault(
            key,
            {
                "identity_key": key,
                "display_name": name,
                "semantic_scholar_author_ids": set(),
                "direct_score": 0.0,
                "second_order_score": 0.0,
                "direct_papers": [],
                "second_order_papers": [],
            },
        )
        item["display_name"] = _prefer_display_name(str(item["display_name"]), name)
        if author.get("authorId"):
            item["semantic_scholar_author_ids"].add(str(author["authorId"]))
        score_key = "direct_score" if tier == "direct" else "second_order_score"
        paper_key = "direct_papers" if tier == "direct" else "second_order_papers"
        item[score_key] += paper_weight
        if len(item[paper_key]) < 6:
            item[paper_key].append(
                {
                    "title": title,
                    "year": paper.get("year"),
                    "seed_title": seed_title or None,
                }
            )


def _listed_label_for_key(key: str, candidate_labels: list[str]) -> str | None:
    for label in candidate_labels:
        if key in _label_keys(label):
            return label
    return None


def _candidate_coauthorship_conflicts(
    author_ids: set[str],
    subject_author_keys: set[str],
    subject_year: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    conflicts: list[dict[str, Any]] = []
    warnings: list[str] = []
    for author_id in sorted(author_ids)[:2]:
        try:
            payload = _request_json(
                _api_url(
                    f"author/{author_id}/papers",
                    {"limit": 1000, "fields": "title,year,authors"},
                ),
                timeout=35,
            )
        except Exception as exc:
            warnings.append(f"Could not verify prior coauthorship for author ID {author_id}: {exc}")
            continue
        papers = payload.get("data") if isinstance(payload.get("data"), list) else []
        for paper in papers:
            year = paper.get("year")
            if subject_year is not None:
                try:
                    if year is None or int(year) >= subject_year:
                        continue
                except (TypeError, ValueError):
                    continue
            coauthor_keys = {_identity_key(str(author.get("name", ""))) for author in _paper_authors(paper)}
            if coauthor_keys & subject_author_keys:
                conflicts.append(
                    {
                        "title": str(paper.get("title", "")),
                        "year": year,
                        "matched_manuscript_author_keys": sorted(coauthor_keys & subject_author_keys),
                    }
                )
            if len(conflicts) >= 5:
                return conflicts, warnings
    return conflicts, warnings


def collect_citation_network(
    underlying_document: dict[str, Any] | None,
    context_note: str,
    candidate_labels: list[str],
    *,
    progress: Callable[[str, list[str] | None], None] | None = None,
    include_outside_candidates: bool = True,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "available": False,
        "provider": "Semantic Scholar Academic Graph API",
        "method": "time-truncated direct and second-order citation reviewer-selection prior",
        "candidates": [],
        "listed_candidates": {},
        "warnings": [],
    }
    try:
        subject = _resolve_subject(underlying_document, context_note)
    except Exception as exc:
        diagnostics["reason"] = str(exc)
        return diagnostics
    subject_id = str(subject.get("paperId", ""))
    if not subject_id:
        diagnostics["reason"] = "The resolved manuscript had no Semantic Scholar paper ID."
        return diagnostics
    try:
        subject_year = int(subject.get("year")) if subject.get("year") is not None else None
    except (TypeError, ValueError):
        subject_year = None
    subject_authors = _paper_authors(subject)
    subject_author_keys = {
        _identity_key(str(author.get("name", ""))) for author in subject_authors
    } - {""}
    diagnostics["subject"] = {
        "paper_id": subject_id,
        "title": subject.get("title"),
        "year": subject_year,
        "authors": [author.get("name") for author in subject_authors],
        "author_records": [
            {
                "name": author.get("name"),
                "author_id": str(author.get("authorId", "")),
            }
            for author in subject_authors
            if author.get("name")
        ],
    }
    if progress:
        progress(
            "Tracing the manuscript citation network",
            [f"Resolved underlying manuscript: {subject.get('title', 'untitled')} ({subject_year or 'year unknown'})."],
        )
    ledger: dict[str, dict[str, Any]] = {}
    try:
        direct_rows = _reference_rows(subject_id, MAX_DIRECT_REFERENCES)
    except Exception as exc:
        diagnostics["reason"] = f"Citation references could not be retrieved: {exc}"
        return diagnostics
    accepted_direct: list[dict[str, Any]] = []
    future_excluded = 0
    for row in direct_rows:
        paper = row.get("citedPaper") if isinstance(row.get("citedPaper"), dict) else {}
        if not paper or not _year_allowed(paper.get("year"), subject_year):
            future_excluded += 1
            continue
        accepted_direct.append(row)
        _add_author_evidence(
            ledger,
            paper,
            tier="direct",
            influential=bool(row.get("isInfluential")),
        )
    direct_seed_rows = sorted(
        accepted_direct,
        key=lambda row: (
            bool(row.get("isInfluential")),
            int((row.get("citedPaper") or {}).get("year") or 0),
        ),
        reverse=True,
    )[:MAX_SECOND_ORDER_SEEDS]
    second_order_count = 0
    for index, row in enumerate(direct_seed_rows):
        seed = row.get("citedPaper") if isinstance(row.get("citedPaper"), dict) else {}
        seed_id = str(seed.get("paperId", ""))
        if not seed_id:
            continue
        try:
            second_rows = _reference_rows(seed_id, MAX_SECOND_ORDER_REFERENCES_PER_SEED)
        except Exception as exc:
            diagnostics["warnings"].append(
                f"Second-order references for {seed.get('title', seed_id)} were unavailable: {exc}"
            )
            continue
        for second_row in second_rows:
            paper = second_row.get("citedPaper") if isinstance(second_row.get("citedPaper"), dict) else {}
            if not paper or not _year_allowed(paper.get("year"), subject_year):
                continue
            second_order_count += 1
            _add_author_evidence(
                ledger,
                paper,
                tier="second_order",
                influential=bool(second_row.get("isInfluential")),
                seed_title=str(seed.get("title", "")),
            )
        if index + 1 < len(direct_seed_rows):
            time.sleep(0.15)
    for key in list(ledger):
        if key in subject_author_keys:
            ledger.pop(key, None)
    items = list(ledger.values())
    for item in items:
        item["raw_prior_score"] = (
            DIRECT_WEIGHT * float(item["direct_score"])
            + SECOND_ORDER_WEIGHT * float(item["second_order_score"])
        )
        item["listed_candidate"] = _listed_label_for_key(item["identity_key"], candidate_labels)
        item["prepublication_coauthor_conflict"] = None
        item["prepublication_coauthor_papers"] = []
        item["relationship_multiplier"] = 1.0
        item["advisor_or_student_relationship"] = "not automatically verified"
    items.sort(key=lambda item: float(item["raw_prior_score"]), reverse=True)
    relationship_targets = [item for item in items if item.get("listed_candidate")]
    if include_outside_candidates:
        relationship_targets += [item for item in items if item not in relationship_targets]
    for item in relationship_targets[:MAX_CANDIDATE_RELATIONSHIP_LOOKUPS]:
        conflicts, warnings = _candidate_coauthorship_conflicts(
            set(item["semantic_scholar_author_ids"]), subject_author_keys, subject_year
        )
        diagnostics["warnings"].extend(warnings)
        item["prepublication_coauthor_conflict"] = bool(conflicts)
        item["prepublication_coauthor_papers"] = conflicts
        if conflicts:
            item["relationship_multiplier"] = PREPUBLICATION_COAUTHOR_MULTIPLIER
        time.sleep(0.1)
    for item in items:
        item["adjusted_prior_score"] = float(item["raw_prior_score"]) * float(
            item["relationship_multiplier"]
        )
    items.sort(key=lambda item: float(item["adjusted_prior_score"]), reverse=True)
    visible_items = (
        items
        if include_outside_candidates
        else [item for item in items if item.get("listed_candidate")]
    )
    maximum = max(
        (float(item["adjusted_prior_score"]) for item in visible_items), default=0.0
    )
    for item in items:
        visible_for_normalization = include_outside_candidates or bool(
            item.get("listed_candidate")
        )
        item["citation_prior_index"] = (
            round(float(item["adjusted_prior_score"]) / maximum, 4)
            if maximum > 0 and visible_for_normalization
            else 0.0
        )
        item["direct_score"] = round(float(item["direct_score"]), 4)
        item["second_order_score"] = round(float(item["second_order_score"]), 4)
        item["raw_prior_score"] = round(float(item["raw_prior_score"]), 4)
        item["adjusted_prior_score"] = round(float(item["adjusted_prior_score"]), 4)
        item["semantic_scholar_author_ids"] = sorted(item["semantic_scholar_author_ids"])
    diagnostics["candidates"] = visible_items[:30]
    diagnostics["outside_candidate_exploration"] = include_outside_candidates
    diagnostics["direct_references_considered"] = len(accepted_direct)
    diagnostics["future_references_excluded"] = future_excluded
    diagnostics["second_order_seed_count"] = len(direct_seed_rows)
    diagnostics["second_order_references_considered"] = second_order_count
    diagnostics["available"] = bool(visible_items)
    if items and not visible_items:
        diagnostics["reason"] = (
            "No supplied candidate was found in the time-truncated citation network; "
            "outside-candidate exploration was disabled."
        )
    diagnostics["identity_note"] = (
        "Initial-plus-surname variants are provisionally grouped for graph counting only; source verification "
        "is required before treating them as one person."
    )
    for label in candidate_labels:
        matches = [item for item in items if item.get("listed_candidate") == label]
        diagnostics["listed_candidates"][label] = (
            {
                "found_in_network": True,
                "citation_prior_index": matches[0]["citation_prior_index"],
                "direct_cited_papers": len(matches[0]["direct_papers"]),
                "second_order_papers": len(matches[0]["second_order_papers"]),
                "prepublication_coauthor_conflict": matches[0]["prepublication_coauthor_conflict"],
                "prepublication_coauthor_papers": matches[0]["prepublication_coauthor_papers"],
                "relationship_multiplier": matches[0]["relationship_multiplier"],
                "advisor_or_student_relationship": matches[0]["advisor_or_student_relationship"],
            }
            if matches
            else {
                "found_in_network": False,
                "citation_prior_index": 0.0,
                "prepublication_coauthor_conflict": None,
                "advisor_or_student_relationship": "not automatically verified",
            }
        )
    if progress and visible_items:
        leaders = " · ".join(
            f"{item['display_name']} ({item['citation_prior_index']:.0%})" for item in visible_items[:5]
        )
        progress(
            "Citation-network candidates ready",
            [
                f"Time-truncated reviewer-selection leads: {leaders}. These are priors, not authorship evidence.",
                f"Excluded {future_excluded} references dated after the manuscript year; prior coauthorship is penalized separately.",
            ],
        )
    return diagnostics


def citation_network_prompt_section(diagnostics: dict[str, Any]) -> str:
    if not diagnostics.get("available"):
        return (
            "No deterministic citation-network packet was available. Reason: "
            + str(diagnostics.get("reason", "unknown"))
        )
    compact_candidates = []
    for item in diagnostics.get("candidates", [])[:18]:
        compact_candidates.append(
            {
                "name": item.get("display_name"),
                "listed_candidate": item.get("listed_candidate"),
                "citation_prior_index": item.get("citation_prior_index"),
                "direct_cited_papers": item.get("direct_papers", [])[:4],
                "second_order_papers": item.get("second_order_papers", [])[:2],
                "prepublication_coauthor_conflict": item.get("prepublication_coauthor_conflict"),
                "prepublication_coauthor_papers": item.get("prepublication_coauthor_papers", [])[:3],
                "relationship_multiplier": item.get("relationship_multiplier"),
                "advisor_or_student_relationship": item.get("advisor_or_student_relationship"),
            }
        )
    packet = {
        "provider": diagnostics.get("provider"),
        "subject": diagnostics.get("subject"),
        "direct_references_considered": diagnostics.get("direct_references_considered"),
        "future_references_excluded": diagnostics.get("future_references_excluded"),
        "second_order_seed_count": diagnostics.get("second_order_seed_count"),
        "listed_candidates": diagnostics.get("listed_candidates"),
        "ranked_network_candidates": compact_candidates,
        "identity_note": diagnostics.get("identity_note"),
        "warnings": diagnostics.get("warnings", [])[:6],
    }
    return json.dumps(packet, ensure_ascii=False, indent=2)
