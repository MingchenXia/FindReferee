from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                    {"candidate": "A", "probability": 0.42},
                    {"candidate": "B", "probability": 0.28},
                    {"candidate": "C", "probability": 0.19},
                    {"candidate": "D", "probability": 0.03},
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
        adjustment = app._apply_review_agreement_adjustment(result, snapshots)
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

    def test_agreement_adjustment_rejects_leader_unsupported_by_all_stylometry_views(self) -> None:
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
        self.assertIn("no support", adjustment["reason"])

    def test_three_of_four_supported_stages_can_outvote_one_review_outlier(self) -> None:
        result = app._normalize_attribution(
            {
                "summary": "Raw adjudication summary.",
                "no_listed_candidate_probability": 0.05,
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.5},
                    {"candidate": "B", "probability": 0.25},
                    {"candidate": "C", "probability": 0.15},
                    {"candidate": "D", "probability": 0.05},
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
        adjustment = app._apply_review_agreement_adjustment(result, snapshots, diagnostics)
        self.assertTrue(adjustment["applied"])
        self.assertEqual(adjustment["supporting_stages"], 3)
        self.assertEqual(adjustment["support_fraction"], 0.75)

    def test_exact_decimal_margin_is_not_rejected_by_float_roundoff(self) -> None:
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
        self.assertTrue(adjustment["applied"])

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


if __name__ == "__main__":
    unittest.main()
