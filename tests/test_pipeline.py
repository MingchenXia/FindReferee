from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import app
import public_corpus
import stylometry


ATOM_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/math/0506021v2</id>
    <title>A legacy solo paper</title>
    <summary>Abstract.</summary>
    <published>2005-06-01T00:00:00Z</published>
    <author><name>Jane Example</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <title>A coauthored paper</title>
    <summary>Abstract.</summary>
    <published>2025-01-01T00:00:00Z</published>
    <author><name>Jane Example</name></author>
    <author><name>Someone Else</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v3</id>
    <title>A same-name mismatch</title>
    <summary>Abstract.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Janet Example</name></author>
  </entry>
</feed>
"""


class PublicCorpusTests(unittest.TestCase):
    def test_followup_corpus_packet_is_compact_and_keeps_work_labels(self) -> None:
        corpora = {
            "Jane Example": [
                {
                    "title": "A long paper",
                    "published": "2020-01-01T00:00:00Z",
                    "abstract_url": "https://arxiv.org/abs/2001.00001v1",
                    "text": "opening " + ("middle prose " * 1_000) + " ending",
                }
            ]
        }
        packet = public_corpus.corpus_followup_section(corpora, characters_per_paper=1_000)
        self.assertIn("CANDIDATE: Jane Example", packet)
        self.assertIn("WORK: A long paper", packet)
        self.assertLess(len(packet), 1_500)

    def test_only_exact_solo_byline_is_accepted(self) -> None:
        records = public_corpus._parse_entries(ATOM_FEED, "Jane Example")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["arxiv_id"], "math/0506021")
        self.assertEqual(records[0]["pdf_url"], "https://arxiv.org/pdf/math/0506021v1")
        self.assertEqual(records[0]["version_used"], "v1")

    def test_alias_normalization_accepts_hyphenation(self) -> None:
        self.assertEqual(
            public_corpus._normalized_name("Jane-Example"),
            public_corpus._normalized_name("JaneExample"),
        )

    def test_query_cache_key_includes_requested_coverage(self) -> None:
        small = public_corpus._query_cache_path("Jane Example", 10)
        broad = public_corpus._query_cache_path("Jane Example", 80)
        self.assertNotEqual(small, broad)

    def test_stale_query_cache_survives_live_refresh_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "query.xml"
            cache_path.write_bytes(ATOM_FEED)
            os.utime(cache_path, (1, 1))
            with (
                patch.object(public_corpus, "_query_cache_path", return_value=cache_path),
                patch.object(public_corpus, "_request_bytes", side_effect=RuntimeError("HTTP 429")),
            ):
                payload, was_cached, warning = public_corpus._query_author("Jane Example", 80)
        self.assertEqual(payload, ATOM_FEED)
        self.assertTrue(was_cached)
        self.assertIn("429", warning or "")


class PipelineTests(unittest.TestCase):
    def test_analyze_endpoint_maps_candidate_scope_switch_safely(self) -> None:
        client = TestClient(app.app)
        with patch.object(
            app, "_perform_analysis", new=AsyncMock(return_value={"ok": True})
        ) as perform:
            response = client.post(
                "/api/analyze",
                data={
                    "mode": "attribution",
                    "candidates": "Jane Example\nAlex Sample",
                    "text_input": "A target referee report.",
                    "explore_outside_candidates": "false",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(perform.await_args.args[5]["disable_outside_candidates"])

        with patch.object(
            app, "_perform_analysis", new=AsyncMock(return_value={"ok": True})
        ) as perform:
            response = client.post(
                "/api/analyze",
                data={
                    "mode": "attribution",
                    "candidates": "",
                    "text_input": "A target referee report.",
                    "explore_outside_candidates": "false",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(perform.await_args.args[5]["disable_outside_candidates"])

    def test_codex_timeout_retries_once_at_configured_recovery_effort(self) -> None:
        attempts: list[list[str]] = []

        def fake_run(args, **kwargs):
            attempts.append(list(args))
            if len(attempts) == 1:
                raise app.subprocess.TimeoutExpired(cmd=args, timeout=1)
            output_path = Path(args[args.index("--output-last-message") + 1])
            output_path.write_text("{}", encoding="utf-8")
            return app.subprocess.CompletedProcess(args, 0, "", "")

        with (
            patch.object(app, "_codex_command", return_value=["codex"]),
            patch.object(app, "CODEX_ENABLE_SEARCH", False),
            patch.object(app, "CODEX_TIMEOUT_SECONDS", 1),
            patch.object(app, "CODEX_TIMEOUT_RETRIES", 1),
            patch.object(app, "CODEX_TIMEOUT_RETRY_EFFORT", "high"),
            patch.object(app.subprocess, "run", side_effect=fake_run),
        ):
            result = app._call_codex(
                "Instructions",
                "Input",
                {"type": "object", "additionalProperties": False, "properties": {}},
                model="gpt-5.6-sol",
                reasoning_effort="xhigh",
            )

        self.assertEqual(len(attempts), 2)
        self.assertIn("model_reasoning_effort=xhigh", attempts[0])
        self.assertIn("model_reasoning_effort=high", attempts[1])
        self.assertEqual(result["_reasoning_effort"], "xhigh (timeout recovery at high)")

    def test_final_timeout_returns_last_complete_review(self) -> None:
        feature_sheet = {
            "sample_diagnostics": "usable",
            "feature_ledger": [],
            "most_discriminative_features": [],
            "features_that_should_be_discounted": [],
        }
        complete_review = {
            "summary": "Complete provisional report.",
            "confidence": "low",
            "no_listed_candidate_probability": 0.1,
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.6},
                {"candidate": "B", "probability": 0.3},
            ],
            "limitations": [],
        }
        with (
            patch.object(app, "ANALYSIS_REVIEW_PASSES", 1),
            patch.object(app, "ADAPTIVE_MAX_TARGETED_ROUNDS", 0),
            patch.object(
                app,
                "_call_model",
                side_effect=[
                    feature_sheet,
                    complete_review,
                    app.HTTPException(
                        status_code=502,
                        detail="The analysis time budget was exhausted.",
                    ),
                ],
            ),
        ):
            result = app._multi_pass_model(
                "Instructions",
                "Prompt",
                "author_attribution",
                {},
                "gpt-test",
                "xhigh",
                feature_source="Target prose",
                deadline_monotonic=app.time.monotonic() + 60,
            )
        self.assertEqual(result["summary"], "Complete provisional report.")
        self.assertTrue(result["_time_budget"]["fallback_used"])
        self.assertTrue(
            any("hard analysis budget" in item for item in result["limitations"])
        )

    def test_missing_independent_distribution_skips_empty_targeted_round(self) -> None:
        feature_sheet = {
            "sample_diagnostics": "usable",
            "feature_ledger": [],
            "most_discriminative_features": [],
            "features_that_should_be_discounted": [],
        }
        recovered = {
            "summary": "Recovered final report.",
            "confidence": "low",
            "no_listed_candidate_probability": 0.1,
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.5},
                {"candidate": "B", "probability": 0.4},
            ],
            "limitations": [],
        }
        progress_events: list[str] = []
        with (
            patch.object(app, "ANALYSIS_REVIEW_PASSES", 1),
            patch.object(app, "ADAPTIVE_MAX_TARGETED_ROUNDS", 2),
            patch.object(
                app,
                "_call_model",
                side_effect=[
                    feature_sheet,
                    app.HTTPException(
                        status_code=502,
                        detail="The independent review timed out.",
                    ),
                    recovered,
                ],
            ) as call_model,
        ):
            result = app._multi_pass_model(
                "Instructions",
                "Prompt",
                "author_attribution",
                {},
                "gpt-test",
                "xhigh",
                progress=lambda stage, _clues=None: progress_events.append(stage),
                feature_source="Target prose",
                deadline_monotonic=app.time.monotonic() + 60,
            )
        self.assertEqual(result["summary"], "Recovered final report.")
        self.assertEqual(call_model.call_count, 3)
        self.assertFalse(any(stage.startswith("Targeted review") for stage in progress_events))
        self.assertIn("recovery adjudication", " ".join(progress_events).lower())
        self.assertEqual(result["_adaptive_review"]["targeted_rounds"], 0)

    def test_attribution_prompt_caps_expertise_and_includes_graph_and_voice_packets(self) -> None:
        prompt = app._attribution_prompt(
            ["A", "B"],
            {"name": "report.txt", "text": "A sufficiently long target report sentence. " * 20, "metadata": {}, "format": "text"},
            "arXiv:2504.21300",
            citation_network_section="CITATION_PACKET_SENTINEL",
            review_voice_section="VOICE_PACKET_SENTINEL",
        )
        self.assertIn("CITATION_PACKET_SENTINEL", prompt)
        self.assertIn("VOICE_PACKET_SENTINEL", prompt)
        self.assertIn("must never exceed 30%", prompt)
        self.assertIn("cannot create a high-confidence or precise identification by themselves", prompt)
        self.assertIn("bounded tie-breaker", prompt)
        self.assertIn("10–15 percentage points", prompt)
        self.assertIn("A positive and a negative report", prompt)
        self.assertIn("A candidate who died before the report could have been written", prompt)
        self.assertIn("Birth year or age by itself must never", prompt)
        self.assertIn("reference_corpus value refers only to user-supplied", prompt)
        self.assertIn("one short event abstract plus one paper does not establish recurrence", prompt)

    def test_outside_candidate_switch_preserves_unnamed_no_match(self) -> None:
        prompt = app._attribution_prompt(
            ["Jane Example / J. Example", "Alex Sample"],
            {
                "name": "report.txt",
                "text": "A sufficiently long target report sentence. " * 20,
                "metadata": {},
                "format": "text",
            },
            "",
            analysis_controls={"disable_outside_candidates": True},
        )
        self.assertIn(
            "Do not search for, name, or propose any outside person", prompt
        )
        self.assertIn("Return outside_candidate_hypotheses as an empty array", prompt)
        self.assertIn("still estimate no_listed_candidate_probability", prompt)
        self.assertNotIn(
            "disabled these analysis dimensions: named discovery", prompt
        )

    def test_listed_source_filter_accepts_slash_aliases(self) -> None:
        self.assertTrue(
            app._source_is_for_listed_candidate(
                {"candidate": "J. Example"}, ["Jane Example / J. Example"]
            )
        )
        self.assertFalse(
            app._source_is_for_listed_candidate(
                {"candidate": "Outside Person"}, ["Jane Example / J. Example"]
            )
        )

    def test_discovery_prompt_uses_citations_as_a_capped_prior(self) -> None:
        prompt = app._discovery_prompt(
            {"name": "report.txt", "text": "Target report text.", "metadata": {}, "format": "text"},
            "arXiv:2504.21300",
            citation_network_section="GRAPH_SEEDS",
        )
        self.assertIn("GRAPH_SEEDS", prompt)
        self.assertIn("Keep this network/academic evidence below 35%", prompt)

    def test_timeline_fields_and_major_progress_animation_are_exposed_in_ui(self) -> None:
        html = (app.APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="profile-birth-year"', html)
        self.assertIn('id="profile-death-year"', html)
        self.assertIn('id="profile-timeline-notes"', html)
        self.assertIn("@keyframes majorDiscovery", html)
        self.assertIn("triggerMajorProgress", html)
        self.assertIn("prefers-reduced-motion", html)

    def test_analysis_elapsed_seconds_is_rounded_and_never_negative(self) -> None:
        self.assertEqual(app._analysis_elapsed_seconds({"created_at": 10.0}, 72.34), 62.3)
        self.assertEqual(app._analysis_elapsed_seconds({"created_at": 72.0}, 10.0), 0.0)

    def test_bounded_review_packet_preserves_late_rounds(self) -> None:
        packet = app._bounded_review_packet(
            [
                {"summary": "FIRST-MARKER " + ("a" * 9_000)},
                {"summary": "LATEST-MARKER " + ("b" * 9_000)},
            ],
            10_000,
        )
        self.assertIn("FIRST-MARKER", packet)
        self.assertIn("LATEST-MARKER", packet)
        self.assertIn("REVIEW REPORT 2 OF 2", packet)

    def test_deterministic_stylometry_returns_three_separate_metrics(self) -> None:
        target = "However, this argument is not complete. Therefore, we should explain the result more carefully. " * 20
        corpora = {
            "A": [{"title": "A1", "text": "However, this proof is not complete. Therefore, we explain it carefully. " * 20}],
            "B": [{"title": "B1", "text": "We calculate equations and obtain values. The computation gives a formula. " * 20}],
        }
        diagnostics = stylometry.build_stylometry_diagnostics(target, corpora)
        self.assertTrue(diagnostics["available"])
        self.assertEqual(set(diagnostics["metric_leaders"]), {
            "character_ngram_best_three_mean", "burrows_delta", "function_word_delta"
        })
        self.assertIn("uncalibrated diagnostics", diagnostics["caveat"])

    def test_short_target_adds_length_matched_character_sensitivity_check(self) -> None:
        target = "However, this argument is not complete. Therefore, explain the result more carefully. " * 5
        corpora = {
            "A": [{"title": "A1", "text": "However, this proof is not complete. Therefore, explain it carefully. " * 80}],
            "B": [{"title": "B1", "text": "We calculate equations and obtain values from the formula. " * 80}],
        }
        diagnostics = stylometry.build_stylometry_diagnostics(target, corpora)
        self.assertTrue(diagnostics["available"])
        self.assertIn("length_matched_character_median", diagnostics["metric_leaders"])
        self.assertIn("length_matched_character", diagnostics["methods"])
        self.assertIn("correlated", diagnostics["methods"]["length_matched_character"])

    def test_surface_language_sample_drops_isolated_math_script_symbols(self) -> None:
        text = (
            "This English referee report discusses a complete metric on Ω and explains why the "
            "curvature argument should be revised. The conclusion is written in ordinary English."
        )
        sample = app._surface_language_sample(text)
        self.assertNotIn("Ω", sample)
        self.assertIn("This English referee report", sample)
        self.assertIn(" a complete metric", sample)
        if app.LanguageDetectorBuilder is not None:
            self.assertIn("English", app._surface_language_detection_note(text))
            greek = (
                "Αυτό είναι ένα ελληνικό κείμενο που περιγράφει ένα μαθηματικό επιχείρημα "
                "και εξηγεί προσεκτικά το συμπέρασμα."
            )
            self.assertIn("Greek", app._surface_language_detection_note(greek))

    def test_deterministic_stylometry_rejects_tiny_target(self) -> None:
        diagnostics = stylometry.build_stylometry_diagnostics(
            "Too short.",
            {"A": [{"text": "long sample " * 100}], "B": [{"text": "other sample " * 100}]},
        )
        self.assertFalse(diagnostics["available"])

    def test_underlying_byline_screen_flags_candidate_alias(self) -> None:
        underlying = {
            "metadata": {"author": "J. Example"},
            "text": "A PAPER TITLE\nJane Example\nAbstract. This paper proves a result.",
        }
        note = app._underlying_candidate_role_note(
            ["Jane Example / J. Example", "Alex Sample"], underlying
        )
        self.assertIn("Jane Example / J. Example", note)
        self.assertIn("strong likely-byline signal", note)
        self.assertIn("before excluding the candidate", note)
        self.assertNotIn("Alex Sample:", note)

    def test_underlying_byline_screen_ignores_late_bibliography_name(self) -> None:
        underlying = {
            "metadata": {},
            "text": "Anonymous draft\n" + ("body " * 2_000) + "Jane Example, References",
        }
        note = app._underlying_candidate_role_note(["Jane Example"], underlying)
        self.assertIn("No candidate name was detected", note)

    def test_underlying_role_screen_flags_comments_on_draft(self) -> None:
        underlying = {
            "metadata": {},
            "text": (
                "Paper body. " * 500
                + "Acknowledgments. The author would like to thank Jane Example for comments on the draft."
            ),
        }
        note = app._underlying_candidate_role_note(["Jane Example", "Alex Sample"], underlying)
        self.assertIn("reviewer-independence counterevidence", note)
        self.assertIn("comments on the draft", note)

    def test_underlying_role_screen_ignores_thanks_to_proposition_near_page_header(self) -> None:
        underlying = {
            "metadata": {},
            "text": (
                "Thanks to Proposition 3.2, it suffices to prove the claim. "
                "14 JANE EXAMPLE The argument now follows from compactness."
            ),
        }
        note = app._underlying_candidate_role_note(["Jane Example"], underlying)
        self.assertNotIn("reviewer-independence counterevidence", note)

    def test_underlying_role_progress_clues_are_compact(self) -> None:
        underlying = {
            "metadata": {"author": "Alex Sample"},
            "text": (
                "Alex Sample\nPaper body. "
                "Acknowledgments. The author would like to thank Jane Example for comments on the draft."
            ),
        }
        clues = app._underlying_role_progress_clues(
            ["Jane Example", "Alex Sample"], underlying
        )
        self.assertEqual(len(clues), 2)
        self.assertTrue(all(len(clue) < 220 for clue in clues))
        self.assertTrue(any("comments on this draft" in clue for clue in clues))
        self.assertTrue(any("Author metadata" in clue for clue in clues))

    def test_manuscript_author_collision_is_exact_and_namesake_safe(self) -> None:
        records, collisions = app._candidate_manuscript_author_leads(
            ["Jane Example / J. Example", "John Example"],
            {},
            None,
            {
                "subject": {
                    "title": "A manuscript",
                    "author_records": [
                        {"name": "Jane Example", "author_id": "123"},
                    ],
                }
            },
            "",
            [],
        )
        self.assertEqual([item["name"] for item in records], ["Jane Example"])
        self.assertEqual(
            [item["candidate"] for item in collisions],
            ["Jane Example / J. Example", "John Example"],
        )

        _, initial_collision = app._candidate_manuscript_author_leads(
            ["J. Example"],
            {},
            None,
            {"subject": {"authors": ["Jane Example"]}},
            "",
            [],
        )
        self.assertEqual([item["candidate"] for item in initial_collision], ["J. Example"])

    def test_report_heading_byline_creates_identity_lead_not_automatic_exclusion(self) -> None:
        records, collisions = app._candidate_manuscript_author_leads(
            ["Jane Example", "Alex Sample"],
            {},
            None,
            {},
            "",
            [],
            {
                "text": (
                    "REPORT FOR ‘A MATHEMATICAL PAPER’ BY JANE EXAMPLE & THIRD AUTHOR\n"
                    "1. Overview\nThe report begins here."
                )
            },
        )
        self.assertEqual([item["name"] for item in records], ["Jane Example"])
        self.assertEqual([item["candidate"] for item in collisions], ["Jane Example"])

    def test_verified_manuscript_author_is_excluded_only_with_identity_source(self) -> None:
        collision = {
            "candidate": "Jane Example",
            "aliases": ["Jane Example"],
            "matched_author_names": ["Jane Example"],
            "lead_evidence": [],
            "candidate_profile": {},
        }
        verified = {
            "matches": [
                {
                    "candidate": "Jane Example",
                    "matched_manuscript_author": "Jane Example",
                    "status": "verified_same_person",
                    "linkage_type": "stable_identifier",
                    "explanation": "The ORCID and official profile match.",
                    "sources": [
                        {
                            "title": "Official profile",
                            "url": "https://example.edu/jane",
                            "identity_evidence": "The profile links the manuscript and ORCID.",
                        },
                        {
                            "title": "Authoritative manuscript byline",
                            "url": "https://doi.org/10.1000/example",
                            "identity_evidence": "The byline and ORCID identify the same Jane Example.",
                        }
                    ],
                }
            ],
            "limitations": [],
        }
        with patch.object(app, "_call_model", return_value=verified):
            screen = app._verify_manuscript_author_identities(
                [{"name": "Jane Example", "source": "test"}],
                [collision],
                {},
                None,
                "",
                "gpt-test",
                "high",
            )
        self.assertEqual(screen["excluded_candidates"], ["Jane Example"])

        no_source = {**verified, "matches": [{**verified["matches"][0], "sources": []}]}
        with patch.object(app, "_call_model", return_value=no_source):
            unresolved = app._verify_manuscript_author_identities(
                [{"name": "Jane Example", "source": "test"}],
                [collision],
                {},
                None,
                "",
                "gpt-test",
                "high",
            )
        self.assertEqual(unresolved["excluded_candidates"], [])
        self.assertEqual(unresolved["unresolved_candidates"], ["Jane Example"])

        name_only = {
            **verified,
            "matches": [
                {
                    **verified["matches"][0],
                    "linkage_type": "name_only",
                }
            ],
        }
        with patch.object(app, "_call_model", return_value=name_only):
            namesake_safe = app._verify_manuscript_author_identities(
                [{"name": "Jane Example", "source": "test"}],
                [collision],
                {},
                None,
                "",
                "gpt-test",
                "high",
            )
        self.assertEqual(namesake_safe["excluded_candidates"], [])
        self.assertEqual(namesake_safe["unresolved_candidates"], ["Jane Example"])

        duplicate_url = {
            **verified,
            "matches": [
                {
                    **verified["matches"][0],
                    "linkage_type": "official_profile_lists_manuscript",
                    "sources": [
                        {
                            **verified["matches"][0]["sources"][0],
                            "identity_evidence": "The profile lists a publication with a matching title.",
                        },
                        {
                            **verified["matches"][0]["sources"][0],
                            "title": "The same profile repeated",
                            "url": "https://example.edu/jane?utm_source=duplicate",
                            "identity_evidence": "The profile lists a publication with a matching title.",
                        },
                    ],
                }
            ],
        }
        with patch.object(app, "_call_model", return_value=duplicate_url):
            duplicate_safe = app._verify_manuscript_author_identities(
                [{"name": "Jane Example", "source": "test"}],
                [collision],
                {},
                None,
                "",
                "gpt-test",
                "high",
            )
        self.assertEqual(duplicate_safe["excluded_candidates"], [])
        self.assertEqual(duplicate_safe["unresolved_candidates"], ["Jane Example"])

    def test_role_exclusion_keeps_manuscript_author_out_of_ranking(self) -> None:
        result = {
            "candidate_evaluations": [
                {"candidate": "Alex Sample", "probability": 0.8},
                {"candidate": "Jane Example", "probability": 0.0},
            ],
            "no_listed_candidate_probability": 0.2,
        }
        screen = {
            "excluded_candidates": ["Jane Example"],
            "matches": [
                {
                    "candidate": "Jane Example",
                    "status": "verified_same_person",
                    "explanation": "Verified by stable identifiers.",
                }
            ],
        }
        restored = app._enforce_manuscript_author_exclusions(result, screen)
        by_name = {item["candidate"]: item for item in restored["candidate_evaluations"]}
        self.assertNotIn("Jane Example", by_name)
        self.assertAlmostEqual(
            sum(item["probability"] for item in restored["candidate_evaluations"])
            + restored["no_listed_candidate_probability"],
            1.0,
        )

    def test_missing_private_corpus_cannot_display_reference_support(self) -> None:
        result = {
            "candidate_evaluations": [
                {
                    "candidate": "Jane Example",
                    "evidence_breakdown": {"reference_corpus": 0.8},
                    "reference_corpus_summary": "Apparent match from public prose.",
                },
                {
                    "candidate": "Alex Sample",
                    "evidence_breakdown": {"reference_corpus": 0.7},
                    "reference_corpus_summary": "Two private reports match.",
                },
            ]
        }
        enforced = app._enforce_private_reference_evidence(
            result,
            {"Alex Sample": [{"text": "private sample"}]},
        )
        by_name = {
            item["candidate"]: item for item in enforced["candidate_evaluations"]
        }
        self.assertEqual(
            by_name["Jane Example"]["evidence_breakdown"]["reference_corpus"],
            0.0,
        )
        self.assertIn(
            "No user-supplied private reference corpus",
            by_name["Jane Example"]["reference_corpus_summary"],
        )
        self.assertEqual(
            by_name["Alex Sample"]["evidence_breakdown"]["reference_corpus"],
            0.7,
        )

    def test_unresolved_manuscript_author_namesake_blocks_precise_claim(self) -> None:
        result = {
            "candidate_evaluations": [
                {"candidate": "Jane Example", "probability": 0.75},
                {"candidate": "Alex Sample", "probability": 0.15},
            ],
            "no_listed_candidate_probability": 0.10,
            "confidence": "high",
            "identity_ambiguity": {"is_ambiguous": False},
            "manuscript_author_screen": {
                "unresolved_candidates": ["Jane Example"]
            },
        }
        outcome = app._attribution_determination(
            result,
            {
                "applied": True,
                "direct_style_families": ["rare errors", "review voice"],
            },
        )
        self.assertEqual(outcome["status"], "unable_to_determine")
        self.assertEqual(outcome["label"], "Identity clarification required")

    def test_long_document_trim_keeps_acknowledgment_tail(self) -> None:
        source = "TITLE " + ("body " * 20_000) + " ACKNOWLEDGMENTS Jane Example"
        trimmed, was_trimmed = app._trim(source)
        self.assertTrue(was_trimmed)
        self.assertIn("TITLE", trimmed)
        self.assertIn("ACKNOWLEDGMENTS Jane Example", trimmed)

    def test_underlying_context_is_bounded_and_keeps_both_ends(self) -> None:
        source = "START" + ("x" * 40_000) + "END"
        excerpt = app._context_excerpt(source, 1_000)
        self.assertLessEqual(len(excerpt), 1_100)
        self.assertTrue(excerpt.startswith("START"))
        self.assertTrue(excerpt.endswith("END"))

    def test_close_candidates_trigger_targeted_review(self) -> None:
        snapshot = app._candidate_consensus_snapshot(
            [
                {
                    "no_listed_candidate_probability": 0.05,
                    "candidate_evaluations": [
                        {"candidate": "A", "probability": 0.44},
                        {"candidate": "B", "probability": 0.40},
                        {"candidate": "C", "probability": 0.11},
                    ],
                }
            ]
        )
        self.assertTrue(app._needs_targeted_review(snapshot))

    def test_unanimous_separated_reviews_receive_bounded_agreement_adjustment(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw adjudication summary.",
                "no_listed_candidate_probability": 0.08,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.42, "evidence_breakdown": {"error_patterns": 0.88}},
                    {"candidate": "B", "probability": 0.28, "evidence_breakdown": {"error_patterns": 0.47}},
                    {"candidate": "C", "probability": 0.19, "evidence_breakdown": {"error_patterns": 0.31}},
                    {"candidate": "D", "probability": 0.03, "evidence_breakdown": {"error_patterns": 0.18}},
                ],
            },
            ["A", "B", "C", "D"],
        )
        snapshots = [
            {
                "stage": f"Review {index}",
                "ranking": [
                    {"candidate": "A", "probability": 0.42},
                    {"candidate": "B", "probability": 0.28},
                    {"candidate": "C", "probability": 0.19},
                    {"candidate": "D", "probability": 0.03},
                ],
                "no_listed_candidate_probability": 0.08,
            }
            for index in range(5)
        ]
        diagnostics = {
            "available": True,
            "metric_leaders": {
                "character_ngram_best_three_mean": "A",
                "burrows_delta": "A",
                "function_word_delta": "B",
            },
        }
        adjustment = app._apply_review_agreement_adjustment(
            result, snapshots, diagnostics, None, {"A": 2}
        )
        self.assertTrue(adjustment["applied"])
        self.assertLessEqual(adjustment["exponent"], 2.0)
        self.assertGreater(result["candidate_evaluations"][0]["probability"], 0.55)

    def test_disagreeing_reviews_do_not_receive_agreement_adjustment(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw adjudication summary.",
                "no_listed_candidate_probability": 0.05,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.48},
                    {"candidate": "B", "probability": 0.47},
                ],
            },
            ["A", "B"],
        )
        snapshots = [
            {"ranking": [{"candidate": "A", "probability": 0.6}, {"candidate": "B", "probability": 0.35}]},
            {"ranking": [{"candidate": "B", "probability": 0.6}, {"candidate": "A", "probability": 0.35}]},
            {"ranking": [{"candidate": "A", "probability": 0.48}, {"candidate": "B", "probability": 0.47}]},
        ]
        adjustment = app._apply_review_agreement_adjustment(result, snapshots)
        self.assertFalse(adjustment["applied"])
        self.assertEqual(result["candidate_evaluations"][0]["probability"], 0.48)

    def test_stylometry_disagreement_is_counterevidence_not_an_absolute_veto(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw adjudication summary.",
                "no_listed_candidate_probability": 0.05,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.55},
                    {"candidate": "B", "probability": 0.25},
                    {"candidate": "C", "probability": 0.15},
                ],
            },
            ["A", "B", "C"],
        )
        snapshots = [
            {
                "ranking": [
                    {"candidate": "A", "probability": 0.55},
                    {"candidate": "B", "probability": 0.25},
                    {"candidate": "C", "probability": 0.15},
                ],
                "no_listed_candidate_probability": 0.05,
            }
            for _ in range(4)
        ]
        diagnostics = {
            "available": True,
            "metric_leaders": {
                "character_ngram_best_three_mean": "B",
                "burrows_delta": "B",
                "function_word_delta": "C",
            },
        }
        adjustment = app._apply_review_agreement_adjustment(result, snapshots, diagnostics)
        self.assertFalse(adjustment["applied"])
        self.assertIn("stylometry_disagreement_note", adjustment)
        self.assertIn("Fewer than two direct", adjustment["reason"])

    def test_three_of_four_supported_stages_can_outvote_one_review_outlier(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw adjudication summary.",
                "no_listed_candidate_probability": 0.05,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.5, "evidence_breakdown": {"error_patterns": 0.86}},
                    {"candidate": "B", "probability": 0.25, "evidence_breakdown": {"error_patterns": 0.48}},
                    {"candidate": "C", "probability": 0.15, "evidence_breakdown": {"error_patterns": 0.32}},
                    {"candidate": "D", "probability": 0.05, "evidence_breakdown": {"error_patterns": 0.22}},
                ],
            },
            ["A", "B", "C", "D"],
        )
        snapshots = []
        for leader in ["B", "A", "A", "A"]:
            ranking = [
                {"candidate": "A", "probability": 0.5 if leader == "A" else 0.25},
                {"candidate": "B", "probability": 0.5 if leader == "B" else 0.25},
                {"candidate": "C", "probability": 0.15},
                {"candidate": "D", "probability": 0.05},
            ]
            snapshots.append({"ranking": ranking, "no_listed_candidate_probability": 0.05})
        diagnostics = {
            "available": True,
            "metric_leaders": {
                "character_ngram_best_three_mean": "A",
                "burrows_delta": "A",
                "function_word_delta": "C",
            },
        }
        adjustment = app._apply_review_agreement_adjustment(
            result, snapshots, diagnostics, None, {"A": 2}
        )
        self.assertTrue(adjustment["applied"])
        self.assertEqual(adjustment["supporting_stages"], 3)
        self.assertEqual(adjustment["support_fraction"], 0.75)

    def test_single_stylometry_view_is_not_a_direct_family_but_private_match_is(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw.",
                "no_listed_candidate_probability": 0.04,
                "candidate_evaluations": [
                    {
                        "candidate": "A",
                        "probability": 0.62,
                        "evidence_breakdown": {
                            "error_patterns": 0.90,
                            "reference_corpus": 0.92,
                        },
                    },
                    {
                        "candidate": "B",
                        "probability": 0.20,
                        "evidence_breakdown": {
                            "error_patterns": 0.42,
                            "reference_corpus": 0.10,
                        },
                    },
                    {"candidate": "C", "probability": 0.14},
                ],
            },
            ["A", "B", "C"],
        )
        snapshots = [
            {
                "ranking": [
                    {"candidate": "A", "probability": 0.62},
                    {"candidate": "B", "probability": 0.20},
                    {"candidate": "C", "probability": 0.14},
                ],
                "no_listed_candidate_probability": 0.04,
            }
            for _ in range(3)
        ]
        diagnostics = {
            "available": True,
            "metric_leaders": {
                "character_ngram_best_three_mean": "B",
                "burrows_delta": "A",
                "function_word_delta": "B",
            },
        }
        adjustment = app._apply_review_agreement_adjustment(
            result, snapshots, diagnostics, None, {"A": 2}
        )
        self.assertTrue(adjustment["applied"])
        self.assertIn("private reference-corpus match", adjustment["direct_style_families"])
        self.assertNotIn("single stylometry view", adjustment["direct_style_families"])

    def test_single_reference_review_voice_cannot_sharpen_probabilities(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw.",
                "no_listed_candidate_probability": 0.05,
                "candidate_evaluations": [
                    {
                        "candidate": "A",
                        "probability": 0.65,
                        "evidence_breakdown": {"error_patterns": 0.90},
                    },
                    {
                        "candidate": "B",
                        "probability": 0.20,
                        "evidence_breakdown": {"error_patterns": 0.40},
                    },
                    {"candidate": "C", "probability": 0.10},
                ],
            },
            ["A", "B", "C"],
        )
        snapshots = [
            {
                "ranking": [
                    {"candidate": "A", "probability": 0.65},
                    {"candidate": "B", "probability": 0.20},
                    {"candidate": "C", "probability": 0.10},
                ],
                "no_listed_candidate_probability": 0.05,
            }
            for _ in range(3)
        ]
        low_data_voice = {
            "available": True,
            "metric_leader": "A",
            "leader_separation": 0.11,
            "reliability": "low",
            "candidates": {
                "A": {
                    "review_like_sample_count": 1,
                    "within_candidate_dispersion": None,
                }
            },
        }
        adjustment = app._apply_review_agreement_adjustment(
            result, snapshots, None, low_data_voice
        )
        self.assertFalse(adjustment["applied"])
        self.assertIn("review_voice_low_data_note", adjustment)

        stable_voice = {
            **low_data_voice,
            "reliability": "moderate",
            "candidates": {
                "A": {
                    "review_like_sample_count": 2,
                    "within_candidate_dispersion": 0.05,
                }
            },
        }
        stable_result = app._normalize_attribution(
            {
                "summary": "Raw.",
                "no_listed_candidate_probability": 0.05,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.65, "evidence_breakdown": {"error_patterns": 0.90}},
                    {"candidate": "B", "probability": 0.20, "evidence_breakdown": {"error_patterns": 0.40}},
                    {"candidate": "C", "probability": 0.10},
                ],
            },
            ["A", "B", "C"],
        )
        stable_adjustment = app._apply_review_agreement_adjustment(
            stable_result, snapshots, None, stable_voice
        )
        self.assertTrue(stable_adjustment["applied"])
        self.assertIn(
            "genre-comparable review voice",
            stable_adjustment["direct_style_families"],
        )

    def test_academic_fit_cannot_substitute_for_direct_style_support(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw adjudication summary.",
                "no_listed_candidate_probability": 0.03,
                "candidate_evaluations": [
                    {
                        "candidate": "A",
                        "probability": 0.41,
                        "evidence_breakdown": {"academic_fit": 0.99},
                    },
                    {
                        "candidate": "B",
                        "probability": 0.31,
                        "evidence_breakdown": {"academic_fit": 0.80},
                    },
                    {"candidate": "C", "probability": 0.18},
                    {"candidate": "D", "probability": 0.07},
                ],
            },
            ["A", "B", "C", "D"],
        )
        snapshots = [
            {
                "ranking": [
                    {"candidate": "A", "probability": 0.41},
                    {"candidate": "B", "probability": 0.31},
                    {"candidate": "C", "probability": 0.18},
                    {"candidate": "D", "probability": 0.07},
                ],
                "no_listed_candidate_probability": 0.03,
            }
            for _ in range(5)
        ]
        diagnostics = {
            "available": True,
            "metric_leaders": {
                "character_ngram_best_three_mean": "B",
                "burrows_delta": "B",
                "function_word_delta": "A",
            },
        }
        adjustment = app._apply_review_agreement_adjustment(result, snapshots, diagnostics)
        self.assertFalse(adjustment["applied"])
        self.assertIn("cannot substitute", adjustment["reason"])
        self.assertAlmostEqual(adjustment["leader_academic_fit_gap"], 0.19)

    def test_targeted_focus_excludes_distant_third_candidate(self) -> None:
        snapshot = {
            "ranked": [("A", 0.43), ("B", 0.34), ("C", 0.14), ("D", 0.05)],
            "top_score": 0.43,
            "second_score": 0.34,
            "no_match": 0.04,
        }
        names, include_no_match = app._targeted_focus(snapshot)
        self.assertEqual(names, ["A", "B"])
        self.assertFalse(include_no_match)

    def test_targeted_focus_prefers_no_match_over_distant_runner_up(self) -> None:
        snapshot = {
            "ranked": [("A", 0.46), ("B", 0.12), ("C", 0.08)],
            "top_score": 0.46,
            "second_score": 0.12,
            "no_match": 0.27,
        }
        names, include_no_match = app._targeted_focus(snapshot)
        self.assertEqual(names, ["A"])
        self.assertTrue(include_no_match)

    def test_clear_leader_stops_targeted_review(self) -> None:
        snapshot = app._candidate_consensus_snapshot(
            [
                {
                    "no_listed_candidate_probability": 0.05,
                    "candidate_evaluations": [
                        {"candidate": "A", "probability": 0.75},
                        {"candidate": "B", "probability": 0.12},
                        {"candidate": "C", "probability": 0.08},
                    ],
                }
            ]
        )
        self.assertFalse(app._needs_targeted_review(snapshot))

    def test_clear_leader_still_receives_one_confirmation_round(self) -> None:
        snapshot = {
            "ranked": [("A", 0.7), ("B", 0.15)],
            "top_score": 0.7,
            "strongest_alternative": 0.15,
            "gap": 0.55,
        }
        self.assertTrue(app._should_run_targeted_review(snapshot, 0))
        self.assertFalse(app._should_run_targeted_review(snapshot, 1))

        close_confirmation = {
            "ranked": [("A", 0.50), ("B", 0.40)],
            "top_score": 0.50,
            "strongest_alternative": 0.40,
            "gap": 0.10,
        }
        self.assertTrue(app._should_run_targeted_review(snapshot, 1, close_confirmation))

    def test_decisive_confirmed_reviews_skip_redundant_final_search(self) -> None:
        reports = [
            {
                "no_listed_candidate_probability": 0.03,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.78},
                    {"candidate": "B", "probability": 0.14},
                ],
            },
            {
                "no_listed_candidate_probability": 0.04,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.72},
                    {"candidate": "B", "probability": 0.18},
                ],
            },
        ]
        self.assertFalse(app._should_enable_final_search("author_attribution", reports, "Public corpus supplied."))

    def test_uncertain_or_disagreeing_reviews_keep_final_search(self) -> None:
        uncertain = [
            {
                "no_listed_candidate_probability": 0.04,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.49},
                    {"candidate": "B", "probability": 0.42},
                ],
            },
            {
                "no_listed_candidate_probability": 0.04,
                "candidate_evaluations": [
                    {"candidate": "B", "probability": 0.48},
                    {"candidate": "A", "probability": 0.43},
                ],
            },
        ]
        self.assertTrue(app._should_enable_final_search("author_attribution", uncertain, "Public corpus supplied."))
        separated_plus_close = [
            {
                "no_listed_candidate_probability": 0.03,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.70},
                    {"candidate": "B", "probability": 0.15},
                ],
            },
            {
                "no_listed_candidate_probability": 0.04,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.50},
                    {"candidate": "B", "probability": 0.40},
                ],
            },
        ]
        self.assertTrue(
            app._should_enable_final_search(
                "author_attribution", separated_plus_close, "Public corpus supplied."
            )
        )
        self.assertTrue(
            app._should_enable_final_search(
                "author_attribution",
                uncertain[:1],
                "No automatically collected public full-text corpus was available",
            )
        )

    def test_close_no_match_alternative_triggers_targeted_review(self) -> None:
        snapshot = app._candidate_consensus_snapshot(
            [
                {
                    "no_listed_candidate_probability": 0.31,
                    "candidate_evaluations": [
                        {"candidate": "A", "probability": 0.45},
                        {"candidate": "B", "probability": 0.12},
                        {"candidate": "C", "probability": 0.07},
                    ],
                }
            ]
        )
        self.assertEqual(snapshot["candidate_gap"], 0.33)
        self.assertAlmostEqual(snapshot["gap"], 0.14)
        self.assertTrue(app._needs_targeted_review(snapshot))

    def test_sub_55_percent_leader_triggers_targeted_review(self) -> None:
        snapshot = app._candidate_consensus_snapshot(
            [
                {
                    "no_listed_candidate_probability": 0.07,
                    "candidate_evaluations": [
                        {"candidate": "A", "probability": 0.50},
                        {"candidate": "B", "probability": 0.28},
                        {"candidate": "C", "probability": 0.15},
                    ],
                }
            ]
        )
        self.assertAlmostEqual(snapshot["gap"], 0.22)
        self.assertTrue(app._needs_targeted_review(snapshot))

    def test_targeted_review_plateau_stops_repetitive_rounds(self) -> None:
        self.assertFalse(app._targeted_review_plateaued(0.03, 0.02, 1))
        self.assertTrue(app._targeted_review_plateaued(0.03, 0.03, 2))
        self.assertTrue(app._targeted_review_plateaued(0.03, 0.01, 2))
        self.assertFalse(app._targeted_review_plateaued(0.03, 0.08, 2))
        self.assertFalse(app._targeted_review_plateaued(0.03, 0.24, 2))

    def test_close_distribution_explicitly_reports_unable_to_determine(self) -> None:
        result = {
            "confidence": "low",
            "no_listed_candidate_probability": 0.08,
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.31},
                {"candidate": "B", "probability": 0.30},
                {"candidate": "C", "probability": 0.29},
            ],
        }
        outcome = app._attribution_determination(result, {"direct_style_families": []})
        self.assertEqual(outcome["status"], "unable_to_determine")
        self.assertAlmostEqual(outcome["margin"], 0.01)

    def test_meaningful_separation_requires_two_direct_families(self) -> None:
        result = {
            "confidence": "high",
            "no_listed_candidate_probability": 0.03,
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.70},
                {"candidate": "B", "probability": 0.17},
                {"candidate": "C", "probability": 0.10},
            ],
        }
        weak = app._attribution_determination(result, {"direct_style_families": ["stylometry"]})
        strong = app._attribution_determination(
            result,
            {
                "applied": True,
                "direct_style_families": ["repeated error fingerprint", "genre-comparable review voice"],
            },
        )
        self.assertEqual(weak["status"], "leading_but_not_precise")
        self.assertEqual(strong["status"], "meaningfully_separated")

        unstable = app._attribution_determination(
            result,
            {
                "applied": False,
                "direct_style_families": ["repeated error fingerprint", "genre-comparable review voice"],
            },
        )
        self.assertEqual(unstable["status"], "leading_but_not_precise")

    def test_outside_candidate_or_identity_ambiguity_prevents_precise_attribution(self) -> None:
        base = {
            "confidence": "high",
            "no_listed_candidate_probability": 0.72,
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.68},
                {"candidate": "B", "probability": 0.12},
            ],
        }
        direct = {"direct_style_families": ["repeated error fingerprint", "genre-comparable review voice"]}
        outside = app._attribution_determination(base, direct)
        self.assertEqual(outside["status"], "unable_to_determine")

        ambiguous = {
            **base,
            "no_listed_candidate_probability": 0.03,
            "identity_ambiguity": {"is_ambiguous": True},
        }
        identity = app._attribution_determination(ambiguous, direct)
        self.assertEqual(identity["status"], "unable_to_determine")


if __name__ == "__main__":
    unittest.main()
