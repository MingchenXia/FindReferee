from __future__ import annotations

import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from unittest.mock import patch

import citation_network


class CitationNetworkTests(unittest.TestCase):
    def test_semantic_scholar_429_is_retried_without_caching_the_error(self) -> None:
        rate_limit = urllib.error.HTTPError(
            "https://api.semanticscholar.org/graph/v1/paper/test",
            429,
            "rate limited",
            {},
            None,
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"paperId":"ok"}'
        with (
            TemporaryDirectory() as directory,
            patch.object(citation_network, "_cache_path", return_value=Path(directory) / "cache.json"),
            patch.object(citation_network.urllib.request, "urlopen", side_effect=[rate_limit, response]),
            patch.object(citation_network.time, "sleep") as sleep,
        ):
            payload = citation_network._request_json(
                "https://api.semanticscholar.org/graph/v1/paper/test"
            )
        self.assertEqual(payload["paperId"], "ok")
        sleep.assert_called_once()

    def test_time_truncated_tiers_aliases_and_relationship_penalty(self) -> None:
        subject = {
            "paperId": "SUBJECT",
            "title": "A Decomposition Lemma",
            "year": 2025,
            "authors": [
                {"authorId": "SU", "name": "Zhitong Su"},
                {"authorId": "ZHANG", "name": "Weijun Zhang"},
            ],
        }
        direct_rows = [
            {
                "isInfluential": True,
                "citedPaper": {
                    "paperId": "MARTA-PAPER",
                    "title": "The Monge-Ampere system",
                    "year": 2023,
                    "authors": [{"authorId": "MARTA", "name": "M. Lewicka"}],
                },
            },
            {
                "isInfluential": False,
                "citedPaper": {
                    "paperId": "FUTURE",
                    "title": "Future leakage",
                    "year": 2026,
                    "authors": [{"authorId": "FUTURE-AUTHOR", "name": "Future Author"}],
                },
            },
            {
                "isInfluential": False,
                "citedPaper": {
                    "paperId": "SELF",
                    "title": "Earlier work by a manuscript author",
                    "year": 2024,
                    "authors": [{"authorId": "SU", "name": "Zhitong Su"}],
                },
            },
        ]
        second_rows = [
            {
                "isInfluential": False,
                "citedPaper": {
                    "paperId": "CAO-PAPER",
                    "title": "Convex integration background",
                    "year": 2019,
                    "authors": [{"authorId": "CAO", "name": "Wentao Cao"}],
                },
            }
        ]

        def references(paper_id: str, _limit: int):
            return direct_rows if paper_id == "SUBJECT" else second_rows if paper_id == "MARTA-PAPER" else []

        def conflicts(author_ids: set[str], _subject_keys: set[str], _subject_year: int | None):
            if "MARTA" in author_ids:
                return ([{"title": "Prior collaboration", "year": 2022}], [])
            return ([], [])

        with (
            patch.object(citation_network, "_resolve_subject", return_value=subject),
            patch.object(citation_network, "_reference_rows", side_effect=references),
            patch.object(citation_network, "_candidate_coauthorship_conflicts", side_effect=conflicts),
            patch.object(citation_network.time, "sleep"),
        ):
            result = citation_network.collect_citation_network(
                None,
                "arXiv:2504.21300",
                ["Marta Lewicka/M. Lewicka", "Wentao Cao"],
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["future_references_excluded"], 1)
        names = {item["display_name"]: item for item in result["candidates"]}
        self.assertNotIn("Future Author", names)
        self.assertNotIn("Zhitong Su", names)
        marta = names["M. Lewicka"]
        cao = names["Wentao Cao"]
        self.assertEqual(marta["listed_candidate"], "Marta Lewicka/M. Lewicka")
        self.assertGreater(marta["raw_prior_score"], cao["raw_prior_score"])
        self.assertAlmostEqual(marta["relationship_multiplier"], 0.22)
        self.assertTrue(
            result["listed_candidates"]["Marta Lewicka/M. Lewicka"]["prepublication_coauthor_conflict"]
        )
        displayed_indices = [item["citation_prior_index"] for item in result["candidates"]]
        self.assertEqual(displayed_indices, sorted(displayed_indices, reverse=True))

    def test_initial_and_full_first_name_share_graph_key_only_provisionally(self) -> None:
        self.assertEqual(
            citation_network._identity_key("Marta Lewicka"),
            citation_network._identity_key("M. Lewicka"),
        )
        self.assertNotEqual(
            citation_network._identity_key("Marta Lewicka"),
            citation_network._identity_key("Anna Lewicka"),
        )

    def test_outside_candidate_switch_keeps_citation_packet_inside_shortlist(self) -> None:
        subject = {
            "paperId": "SUBJECT",
            "title": "A manuscript",
            "year": 2025,
            "authors": [{"authorId": "SUBJECT-AUTHOR", "name": "Paper Author"}],
        }
        direct_rows = [
            {
                "isInfluential": False,
                "citedPaper": {
                    "paperId": "LISTED-PAPER",
                    "title": "Listed work",
                    "year": 2020,
                    "authors": [{"authorId": "LISTED", "name": "Listed Person"}],
                },
            },
            {
                "isInfluential": True,
                "citedPaper": {
                    "paperId": "OUTSIDE-PAPER",
                    "title": "Outside work",
                    "year": 2021,
                    "authors": [{"authorId": "OUTSIDE", "name": "Outside Person"}],
                },
            },
        ]

        def references(paper_id: str, _limit: int):
            return direct_rows if paper_id == "SUBJECT" else []

        with (
            patch.object(citation_network, "_resolve_subject", return_value=subject),
            patch.object(citation_network, "_reference_rows", side_effect=references),
            patch.object(
                citation_network,
                "_candidate_coauthorship_conflicts",
                return_value=([], []),
            ) as relationship_check,
            patch.object(citation_network.time, "sleep"),
        ):
            result = citation_network.collect_citation_network(
                None,
                "arXiv:2501.00001",
                ["Listed Person"],
                include_outside_candidates=False,
            )

        self.assertTrue(result["available"])
        self.assertFalse(result["outside_candidate_exploration"])
        self.assertEqual(
            [item["display_name"] for item in result["candidates"]],
            ["Listed Person"],
        )
        self.assertEqual(result["candidates"][0]["citation_prior_index"], 1.0)
        relationship_check.assert_called_once()
        checked_author_ids = relationship_check.call_args.args[0]
        self.assertEqual(checked_author_ids, {"LISTED"})


if __name__ == "__main__":
    unittest.main()
