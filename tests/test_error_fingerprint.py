from __future__ import annotations

import unittest
from unittest.mock import patch

import error_fingerprint


TARGET = (
    "The authors claim the estimate occured in Section 2; this is not neccessary. The behaviour of the "
    "plurisubharmonic weight is seperate from the main argu-\nment, and the informations given are thin. "
    "Kähler metrics and the ﬁnite cohomology are fine."
)
CORPORA = {
    "A": [
        {"text": "It occured to us that the behaviour is neccessary."},
        {"text": "This occured again; the informations are neccessary and the cohomology is plurisubharmonic."},
    ],
    "B": [
        {"text": "The behaviour of the cohomology in the plurisubharmonic setting."},
        {"text": "More behaviour on cohomology in a seperate plurisubharmonic topic."},
    ],
    "C": [{"text": "Clean prose with nothing unusual in it at all."}],
}


@unittest.skipIf(error_fingerprint.SpellChecker is None, "pyspellchecker is not installed")
class ErrorFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = error_fingerprint.build_error_fingerprint_diagnostics(TARGET, CORPORA)
        self.shared = {
            name: {item["token"]: item for item in details["shared_fingerprints"]}
            for name, details in self.result["candidates"].items()
        }

    def test_exclusive_recurring_misspellings_identify_the_leader(self) -> None:
        self.assertEqual(self.result["leader"], "A")
        self.assertEqual(self.result["candidates"]["A"]["exclusive_misspelling_like"], 2)
        occured = self.shared["A"]["occured"]
        self.assertEqual(occured["class"], "misspelling_like")
        self.assertEqual(occured["suggested_form"], "occurred")
        self.assertEqual(occured["candidate_works"], 2)
        self.assertEqual(occured["other_candidates_using_it"], 0)

    def test_a_token_from_a_single_work_is_not_a_fingerprint(self) -> None:
        self.assertNotIn("informations", self.shared["A"])
        self.assertNotIn("seperate", self.shared["B"])

    def test_conventions_and_field_terms_are_not_called_errors(self) -> None:
        self.assertEqual(self.shared["B"]["behaviour"]["class"], "variant_spelling")
        self.assertEqual(self.shared["B"]["behaviour"]["suggested_form"], "behavior")
        self.assertEqual(self.shared["B"]["cohomology"]["class"], "unrecognized_term")
        self.assertEqual(self.shared["B"]["cohomology"]["other_candidates_using_it"], 1)
        self.assertEqual(self.result["candidates"]["B"]["exclusive_misspelling_like"], 0)

    def test_names_and_pdf_artifacts_are_not_counted(self) -> None:
        listed = {item["token"] for item in self.result["target_nonstandard_tokens"]}
        self.assertNotIn("kahler", listed)
        self.assertNotIn("argu", listed)
        self.assertNotIn("nite", listed)

    def test_missing_lexicon_reports_unavailable(self) -> None:
        with patch.object(error_fingerprint, "_lexicon", return_value=None):
            self.assertFalse(error_fingerprint.build_error_fingerprint_diagnostics(TARGET, CORPORA)["available"])


if __name__ == "__main__":
    unittest.main()
