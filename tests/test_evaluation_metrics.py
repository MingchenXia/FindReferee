from __future__ import annotations

import math
import unittest

import evaluation_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_case_metrics_reward_correct_separated_result(self) -> None:
        result = {
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.70},
                {"candidate": "B", "probability": 0.20},
            ],
            "no_listed_candidate_probability": 0.10,
        }
        score = evaluation_metrics.score_case(result, "A")
        self.assertTrue(score["top1_correct"])
        self.assertTrue(score["decisive_correct"])
        self.assertAlmostEqual(score["true_class_margin"], 0.50)
        self.assertAlmostEqual(score["log_loss"], -math.log(0.70))

    def test_log_loss_and_brier_penalize_confident_wrong_result(self) -> None:
        good = evaluation_metrics.score_case(
            {
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.8},
                    {"candidate": "B", "probability": 0.1},
                ],
                "no_listed_candidate_probability": 0.1,
            },
            "A",
        )
        bad = evaluation_metrics.score_case(
            {
                "candidate_evaluations": [
                    {"candidate": "A", "probability": 0.1},
                    {"candidate": "B", "probability": 0.8},
                ],
                "no_listed_candidate_probability": 0.1,
            },
            "A",
        )
        self.assertLess(good["log_loss"], bad["log_loss"])
        self.assertLess(good["brier_score"], bad["brier_score"])

    def test_top1_requires_a_unique_leader_and_reports_ties_separately(self) -> None:
        result = {
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.40},
                {"candidate": "B", "probability": 0.40},
            ],
            "no_listed_candidate_probability": 0.20,
            "determination": {"status": "unable_to_determine"},
        }
        score = evaluation_metrics.score_case(result, "A")
        self.assertFalse(score["top1_correct"])
        self.assertTrue(score["top1_including_ties"])
        self.assertTrue(score["top1_tie"])
        self.assertEqual(score["expected_rank"], 1)

        aggregate = evaluation_metrics.aggregate_scores([score])
        self.assertEqual(aggregate["top1_accuracy"], 0.0)
        self.assertEqual(aggregate["top1_including_ties_accuracy"], 1.0)
        self.assertEqual(aggregate["top1_tie_rate"], 1.0)

    def test_js_stability_handles_different_label_sets(self) -> None:
        divergence = evaluation_metrics.jensen_shannon_divergence(
            {"A": 0.8, "B": 0.2}, {"A": 0.8, "C": 0.2}
        )
        self.assertGreater(divergence, 0)
        self.assertLessEqual(divergence, math.log(2))

    def test_explicit_non_precise_status_overrides_large_numeric_gap(self) -> None:
        result = {
            "candidate_evaluations": [
                {"candidate": "A", "probability": 0.80},
                {"candidate": "B", "probability": 0.10},
            ],
            "no_listed_candidate_probability": 0.10,
            "determination": {"status": "leading_but_not_precise"},
        }
        score = evaluation_metrics.score_case(result, "A")
        self.assertTrue(score["top1_correct"])
        self.assertFalse(score["precise_claim"])
        self.assertFalse(score["decisive_correct"])

    def test_selective_metrics_penalize_wrong_precise_claim_without_penalizing_abstention(self) -> None:
        rows = [
            evaluation_metrics.score_case(
                {
                    "candidate_evaluations": [
                        {"candidate": "A", "probability": 0.75},
                        {"candidate": "B", "probability": 0.15},
                    ],
                    "no_listed_candidate_probability": 0.10,
                    "determination": {"status": "meaningfully_separated"},
                },
                "A",
            ),
            evaluation_metrics.score_case(
                {
                    "candidate_evaluations": [
                        {"candidate": "B", "probability": 0.75},
                        {"candidate": "A", "probability": 0.15},
                    ],
                    "no_listed_candidate_probability": 0.10,
                    "determination": {"status": "meaningfully_separated"},
                },
                "A",
            ),
            evaluation_metrics.score_case(
                {
                    "candidate_evaluations": [
                        {"candidate": "A", "probability": 0.40},
                        {"candidate": "B", "probability": 0.38},
                    ],
                    "no_listed_candidate_probability": 0.22,
                    "determination": {"status": "unable_to_determine"},
                },
                "A",
            ),
        ]
        aggregate = evaluation_metrics.aggregate_scores(rows)
        self.assertAlmostEqual(aggregate["precise_claim_coverage"], 2 / 3)
        self.assertAlmostEqual(aggregate["precise_claim_precision"], 0.5)
        self.assertAlmostEqual(aggregate["false_precise_claim_rate"], 1 / 3)
        self.assertAlmostEqual(aggregate["unable_to_determine_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
