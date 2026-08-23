"""Conservative retrieval of public, solo-authored arXiv comparison samples."""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
USER_AGENT = "AuthorAttribution/0.1 (local noncommercial research tool)"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
QUERY_CACHE_SECONDS = 7 * 24 * 60 * 60
MAX_PDF_BYTES = 12 * 1024 * 1024
MAX_EXCERPT_CHARS = 7_000
PROMPT_EXCERPT_CHARS = max(
    2_500, min(MAX_EXCERPT_CHARS, int(os.getenv("AUTHOR_ATTRIBUTION_PUBLIC_EXCERPT_CHARS", "2500")))
)


def _cache_root() -> Path:
    configured = os.getenv("AUTHOR_ATTRIBUTION_PUBLIC_CACHE", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.home() / ".cache" / "author-attribution" / "arxiv",
        Path(tempfile.gettempdir()) / "author-attribution-public-corpus",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("No writable public-corpus cache directory is available.")


def _normalized_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized.casefold() if char.isalnum())


def _aliases(label: str) -> list[str]:
    aliases = [item.strip() for item in label.split("/") if item.strip()]
    return aliases or [label.strip()]


def _request_bytes(url: str, timeout: int, maximum: int | None = None) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").casefold() not in ARXIV_HOSTS:
        raise ValueError("Only HTTPS URLs on official arXiv hosts are allowed.")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if maximum is None:
            return response.read()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(256 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ValueError(f"Public PDF exceeds the {maximum // (1024 * 1024)} MB safety limit.")
        return b"".join(chunks)


def _query_cache_path(label: str, max_results: int) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "-", _normalized_name(label))[:80] or "candidate"
    return _cache_root() / f"query-{safe}-{max_results}.xml"


def _query_author(label: str, max_results: int) -> tuple[bytes, bool, str | None]:
    cache_path = _query_cache_path(label, max_results)
    if cache_path.is_file() and time.time() - cache_path.stat().st_mtime < QUERY_CACHE_SECONDS:
        return cache_path.read_bytes(), True, None
    author_query = " OR ".join(f'au:"{name}"' for name in _aliases(label))
    parameters = urllib.parse.urlencode(
        {
            "search_query": author_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    try:
        payload = _request_bytes(f"{ARXIV_API}?{parameters}", timeout=45)
    except Exception as exc:
        if cache_path.is_file():
            # Historical v1 solo-paper metadata remains useful when arXiv briefly
            # rate-limits refreshes. Reuse it instead of discarding the already
            # downloaded full-text corpus and silently degrading attribution.
            payload = cache_path.read_bytes()
            try:
                cache_path.touch()
            except OSError:
                pass
            return payload, True, f"Live metadata refresh failed; reused cached records: {exc}"
        raise
    cache_path.write_bytes(payload)
    return payload, False, None


def _entry_text(entry: ET.Element, tag: str) -> str:
    return re.sub(r"\s+", " ", entry.findtext(f"atom:{tag}", default="", namespaces=ATOM)).strip()


def _parse_entries(payload: bytes, label: str) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    accepted_names = {_normalized_name(name) for name in _aliases(label)}
    records: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM):
        authors = [
            re.sub(r"\s+", " ", author.findtext("atom:name", default="", namespaces=ATOM)).strip()
            for author in entry.findall("atom:author", ATOM)
        ]
        if len(authors) != 1 or _normalized_name(authors[0]) not in accepted_names:
            continue
        identifier = _entry_text(entry, "id")
        identifier_path = urllib.parse.urlparse(identifier).path
        if not identifier_path.startswith("/abs/"):
            continue
        versioned_id = identifier_path.removeprefix("/abs/").strip("/")
        if not versioned_id:
            continue
        base_id = re.sub(r"v\d+$", "", versioned_id)
        published = _entry_text(entry, "published")
        try:
            published_year = int(published[:4])
        except (TypeError, ValueError):
            published_year = 9999
        records.append(
            {
                "arxiv_id": base_id,
                "title": _entry_text(entry, "title"),
                "summary": _entry_text(entry, "summary"),
                "authors": authors,
                "published": published,
                "published_year": published_year,
                "abstract_url": f"https://arxiv.org/abs/{base_id}v1",
                "pdf_url": f"https://arxiv.org/pdf/{base_id}v1",
                "version_used": "v1",
            }
        )
    return records


def _distributed_selection(records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    historical = [record for record in records if record["published_year"] <= 2025]
    if len(historical) <= maximum:
        return historical
    # Include recent, middle, and older original prose rather than four adjacent papers.
    positions = [round(index * (len(historical) - 1) / (maximum - 1)) for index in range(maximum)]
    return [historical[position] for position in positions]


def _sample_body(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    normalized = re.sub(r"[ \t]+", " ", text.replace("\x00", ""))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if len(normalized) <= limit:
        return normalized
    window = max(150, limit // 5)
    usable_end = max(window, int(len(normalized) * 0.78))
    starts = [int(index * max(0, usable_end - window) / 4) for index in range(5)]
    pieces = []
    for start in starts:
        end = min(len(normalized), start + window)
        piece = normalized[start:end]
        if start:
            newline = piece.find("\n")
            if 0 <= newline < 180:
                piece = piece[newline + 1 :]
        pieces.append(piece.strip())
    return "\n\n[Excerpt jump]\n\n".join(pieces)[:limit]


def _paper_cache_path(arxiv_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", arxiv_id)
    return _cache_root() / f"paper-{safe}-v1.json"


def _paper_excerpt(record: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    cache_path = _paper_cache_path(record["arxiv_id"])
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("text"):
                return cached, True
        except (OSError, json.JSONDecodeError):
            pass
    payload = _request_bytes(record["pdf_url"], timeout=60, maximum=MAX_PDF_BYTES)
    if not payload.startswith(b"%PDF"):
        raise ValueError("arXiv did not return a readable PDF.")
    reader = PdfReader(io.BytesIO(payload))
    page_count = len(reader.pages)
    if page_count <= 45:
        page_indexes = range(page_count)
    else:
        page_indexes = sorted(
            set(range(18))
            | set(range(max(18, page_count // 2 - 4), min(page_count, page_count // 2 + 4)))
            | set(range(max(0, page_count - 10), page_count))
        )
    extracted = "\n\n".join(reader.pages[index].extract_text() or "" for index in page_indexes)
    sample = {
        **record,
        "text": _sample_body(extracted),
        "pages": page_count,
        "pages_sampled": len(list(page_indexes)),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_path.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")
    return sample, False


def collect_arxiv_corpora(
    candidate_labels: list[str],
    *,
    max_results_per_candidate: int = 80,
    max_full_text_papers_per_candidate: int = 4,
    progress: Callable[[str, list[str] | None], None] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return exact-name solo-paper excerpts and transparent retrieval diagnostics.

    Failures are isolated per candidate. The caller can continue with model-led web
    research when arXiv has no matching solo work or the network is unavailable.
    """
    corpora: dict[str, list[dict[str, Any]]] = {label: [] for label in candidate_labels}
    diagnostics: dict[str, Any] = {"provider": "arXiv public API", "candidates": {}, "errors": []}
    previous_live_query = False
    for label in candidate_labels:
        if progress:
            progress(f"Collecting public solo works for {label}", None)
        try:
            if previous_live_query:
                time.sleep(3)
            payload, query_cached, query_warning = _query_author(label, max_results_per_candidate)
            previous_live_query = not query_cached
            records = _parse_entries(payload, label)
            selected = _distributed_selection(records, max_full_text_papers_per_candidate)
            samples = []
            paper_errors = []
            cached_papers = 0
            for record in selected:
                try:
                    sample, was_cached = _paper_excerpt(record)
                    samples.append(sample)
                    cached_papers += int(was_cached)
                except Exception as exc:  # A single inaccessible PDF should not abort attribution.
                    paper_errors.append(f"{record['arxiv_id']}: {exc}")
            corpora[label] = samples
            diagnostics["candidates"][label] = {
                "exact_solo_metadata_matches": len(records),
                "pre_2026_solo_metadata_matches": sum(1 for item in records if item["published_year"] <= 2025),
                "full_text_samples": len(samples),
                "cached_full_text_samples": cached_papers,
                "query_cached": query_cached,
                "query_warning": query_warning,
                "paper_errors": paper_errors,
            }
            if progress:
                progress(
                    f"Public solo-work corpus ready for {label}",
                    [
                        f"{label}: {len(samples)} original solo-paper excerpts; "
                        f"{sum(1 for item in records if item['published_year'] <= 2025)} pre-2026 solo records found."
                    ],
                )
        except Exception as exc:
            diagnostics["errors"].append(f"{label}: {exc}")
            diagnostics["candidates"][label] = {
                "exact_solo_metadata_matches": 0,
                "pre_2026_solo_metadata_matches": 0,
                "full_text_samples": 0,
                "cached_full_text_samples": 0,
                "query_cached": False,
                "paper_errors": [str(exc)],
            }
    return corpora, diagnostics


def corpus_prompt_section(corpora: dict[str, list[dict[str, Any]]], diagnostics: dict[str, Any]) -> str:
    blocks = []
    for label, samples in corpora.items():
        if not samples:
            blocks.append(f"CANDIDATE: {label}\nNo exact-name solo-authored arXiv full-text sample was available.")
            continue
        paper_blocks = []
        for sample in samples:
            prompt_excerpt = _sample_body(sample["text"], PROMPT_EXCERPT_CHARS)
            paper_blocks.append(
                f"""PUBLIC SOLO WORK: {sample['title']}
Author: {sample['authors'][0]}
Original submission: {sample['published']}
Version used: {sample['version_used']} (historical original)
Source: {sample['abstract_url']}
---
{prompt_excerpt}
---"""
            )
        blocks.append(f"CANDIDATE: {label}\n" + "\n\n".join(paper_blocks))
    return (
        "Official public-corpus diagnostics:\n"
        + json.dumps(diagnostics, ensure_ascii=False)
        + "\n\n"
        + "\n\n".join(blocks)
    )


def corpus_followup_section(
    corpora: dict[str, list[dict[str, Any]]], *, characters_per_paper: int = 1_000
) -> str:
    """Keep small distributed raw-text windows available to focused review rounds."""
    blocks: list[str] = []
    for label, samples in corpora.items():
        paper_blocks: list[str] = []
        for sample in samples:
            paper_blocks.append(
                f"""WORK: {sample.get('title', 'Untitled')}
Original submission: {sample.get('published', 'unknown')}
Source: {sample.get('abstract_url', '')}
---
{_sample_body(str(sample.get('text', '')), characters_per_paper)}
---"""
            )
        blocks.append(
            f"CANDIDATE: {label}\n"
            + ("\n\n".join(paper_blocks) if paper_blocks else "No raw public sample was available.")
        )
    return "\n\n".join(blocks)
