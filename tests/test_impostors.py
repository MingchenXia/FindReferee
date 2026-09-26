from __future__ import annotations

import random
import unittest

import impostors


def _author(seed: int):
    """A synthetic author: a private vocabulary, word preferences, and punctuation habit."""
    generator = random.Random(seed)
    vocabulary = ["".join(generator.choice("abcdefghilmnoprstu") for _ in range(generator.randint(2, 8))) for _ in range(250)]
    weights = [generator.random() ** 3 for _ in vocabulary]
    punctuation = generator.choice([", ", "; ", " - ", ": "])

    def write(words: int) -> str:
        chosen = generator.choices(vocabulary, weights, k=words)
        return " ".join(word + (punctuation if index % 9 == 0 else "") for index, word in enumerate(chosen))

    return write


class GeneralImpostorsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a, self.b, self.c, self.d = (_author(seed) for seed in (1, 2, 3, 4))
        self.candidates = {
            "A": [{"text": self.a(1_200)} for _ in range(3)],
            "B": [{"text": self.b(1_200)} for _ in range(3)],
        }
        self.impostors = {
            "C": [{"text": self.c(1_200)} for _ in range(3)],
            "D": [{"text": self.d(1_200)} for _ in range(3)],
        }

    def test_listed_author_beats_outside_impostors(self) -> None:
        result = impostors.build_impostor_diagnostics(self.a(900), self.candidates, self.impostors)
        self.assertTrue(result["available"])
        self.assertEqual(result["leader"], "A")
        self.assertGreaterEqual(result["candidates"]["A"]["verification_score"], 0.9)
        self.assertLessEqual(result["external_impostor_win_rate"], 0.1)
        self.assertEqual(result["external_impostor_authors"], ["C", "D"])

    def test_unlisted_author_raises_the_external_win_rate(self) -> None:
        result = impostors.build_impostor_diagnostics(self.c(900), self.candidates, self.impostors)
        self.assertGreaterEqual(result["external_impostor_win_rate"], 0.9)
        self.assertLessEqual(result["candidates"]["A"]["verification_score"], 0.1)

    def test_result_is_reproducible(self) -> None:
        target = self.a(900)
        self.assertEqual(
            impostors.build_impostor_diagnostics(target, self.candidates, self.impostors),
            impostors.build_impostor_diagnostics(target, self.candidates, self.impostors),
        )

    def test_without_outside_impostors_it_only_measures_ranking_stability(self) -> None:
        result = impostors.build_impostor_diagnostics(self.a(900), self.candidates)
        self.assertEqual(result["reliability"], "closed set only")
        self.assertIsNone(result["external_impostor_win_rate"])
        self.assertIsNone(result["candidates"]["A"]["verification_score"])

    def test_listed_name_is_never_reused_as_an_impostor(self) -> None:
        result = impostors.build_impostor_diagnostics(
            self.a(900), self.candidates, {"A": self.candidates["A"], **self.impostors}
        )
        self.assertEqual(result["external_impostor_authors"], ["C", "D"])

    def test_tiny_target_is_rejected(self) -> None:
        self.assertFalse(impostors.build_impostor_diagnostics("Too short.", self.candidates, self.impostors)["available"])


if __name__ == "__main__":
    unittest.main()
