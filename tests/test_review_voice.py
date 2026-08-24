from __future__ import annotations

import unittest

import review_voice


class ReviewVoiceTests(unittest.TestCase):
    def test_paper_prose_is_not_mislabeled_as_review_voice(self) -> None:
        text = (
            "We prove the main theorem by constructing an elliptic system and applying a compactness argument. "
            "The resulting estimate implies convergence of the sequence. The proof is divided into three steps. "
            "First we establish uniform bounds. Next we pass to the weak limit. Finally we identify the limit "
            "and obtain the required regularity for every solution in the prescribed class."
        )
        profile = review_voice.extract_review_voice(text)
        self.assertFalse(profile["review_like"])

    def test_review_voice_separates_different_critical_stances(self) -> None:
        target = (
            "The paper contains an interesting improvement. However, by my judgement the main result is not "
            "enough for this journal. I think the authors should explain why the additional construction is "
            "necessary. The so called generalisation also has a problem: the argument does not appear new. "
            "I therefore cannot recommend publication in the present form."
        )
        corpora = {
            "Candidate A": [
                {
                    "name": "Known quick opinion",
                    "text": (
                        "I find the estimate interesting. However, in my judgement it is not sufficient for "
                        "this journal. The authors should clarify the so called improvement. I therefore do "
                        "not recommend publication in the present form."
                    ),
                }
            ],
            "Candidate B": [
                {
                    "name": "Known formal report",
                    "text": (
                        "This manuscript studies a relevant problem and presents several useful lemmas. The "
                        "article is clearly organized. Could the authors add references and explain the scope "
                        "of Theorem 2? Subject to these minor revisions, the paper may be suitable for publication."
                    ),
                }
            ],
        }
        result = review_voice.build_review_voice_diagnostics(target, corpora)
        self.assertTrue(result["available"])
        self.assertEqual(result["metric_leader"], "Candidate A")
        self.assertGreater(result["leader_separation"], 0.035)


if __name__ == "__main__":
    unittest.main()
