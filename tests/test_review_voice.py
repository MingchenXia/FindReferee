from __future__ import annotations

import re
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


    def test_single_pass_feature_counts_equal_per_pattern_counts(self) -> None:
        phrases = (
            "I my me we our us the author the authors the manuscript the paper the article may might could "
            "seem seems appear appears perhaps possibly in my view in my opinion in my judgment in my judgement "
            "interesting important novel valuable useful clear clearly well-written well written improvement "
            "concern concerns problem problems issue issues flaw flaws incorrect insufficient not clear "
            "not convincing not enough not new not original not correct not suitable lack lacks weak weakness "
            "weaknesses recommend recommendation recommended recommending accept acceptance accepted reject "
            "rejection rejected publish publishable published publishing suitable for journal major revision "
            "minor revision should must need to needs to needed to I suggest I ask I encourage it would be "
            "useful it would be helpful it would be better however nevertheless although while on the other "
            "hand but reference references cite cited cites citation citations literature previous work"
        )
        text = "\n".join(
            [phrases, phrases.upper(), phrases.replace(" ", ", "), "Clearly; the authors (i) must... I? Me!"]
        )
        for name, patterns in review_voice._PATTERNS.items():
            with self.subTest(feature=name):
                expected = sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
                self.assertGreater(expected, 0)
                self.assertEqual(len(review_voice._FEATURE_REGEXES[name].findall(text)), expected)

if __name__ == "__main__":
    unittest.main()
