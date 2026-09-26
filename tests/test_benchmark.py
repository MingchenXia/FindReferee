from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import app
import benchmark


RESULT = {
    "summary": "Report.",
    "model": "gpt-test",
    "review_strategy": "multi-pass review",
    "candidate_evaluations": [
        {"candidate": "Alice Author", "probability": 0.6},
        {"candidate": "Bob Writer", "probability": 0.3},
    ],
    "no_listed_candidate_probability": 0.1,
    "determination": {"status": "leading_but_not_precise"},
}


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name, expected, status in (("alpha", "Alice Author", "confirmed"), ("beta", "Bob Writer", "belief")):
            folder = self.root / name
            folder.mkdir()
            (folder / "reference.txt").write_text("A known report by Alice. " * 30, encoding="utf-8")
            (folder / "case.json").write_text(
                json.dumps(
                    {
                        "target_text": "The authors should clarify the estimate. " * 20,
                        "candidates": ["Alice Author", "Bob Writer"],
                        "context": f"context for {name}",
                        "references": {"Alice Author": ["reference.txt"]},
                        "expected_author": expected,
                        "label_status": status,
                    }
                ),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _main(self, *argv: str) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(benchmark.main(["--root", str(self.root), *argv]), 0)
        return output.getvalue()

    def test_run_passes_inputs_only_and_saves_each_repetition(self) -> None:
        with patch.object(app, "_perform_analysis", new=AsyncMock(return_value=dict(RESULT))) as analysis:
            self._main("run", "--repeat", "2", "--model", "gpt-test", "--effort", "high")
        self.assertEqual(analysis.await_count, 4)
        for call in analysis.await_args_list:
            mode, candidates, documents, context, _profiles, _controls, model, effort, references = call.args[:9]
            self.assertEqual((mode, candidates, model, effort), ("attribution", ["Alice Author", "Bob Writer"], "gpt-test", "high"))
            self.assertTrue(context.startswith("context for"))
            self.assertEqual(len(references["Alice Author"]), 1)
            self.assertNotIn("expected_author", repr(call.args))
            self.assertNotIn("label_status", repr(call.args))
        self.assertEqual(len(list((self.root / "alpha" / "runs").glob("*.json"))), 2)

    def test_score_and_calibration_use_the_withheld_labels(self) -> None:
        with patch.object(app, "_perform_analysis", new=AsyncMock(return_value=dict(RESULT))):
            self._main("run", "--repeat", "2", "--model", "gpt-test")
        report = json.loads(self._main("score", "--json"))
        self.assertEqual(report["aggregate"]["case_count"], 2)
        by_case = {row["case"]: row for row in report["cases"]}
        self.assertTrue(by_case["alpha"]["top1_correct"])
        self.assertFalse(by_case["beta"]["top1_correct"])
        every_run = json.loads(self._main("score", "--json", "--runs", "all", "--confirmed-only"))
        self.assertEqual({row["case"] for row in every_run["cases"]}, {"alpha"})
        self.assertEqual(every_run["repeated_run_stability"]["alpha"]["run_count"], 2)

        output = self.root / "calibration.json"
        self._main("fit-calibration", "--output", str(output))
        calibration = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(calibration["model"], "gpt-test")
        self.assertEqual((calibration["case_count"], calibration["run_count"]), (2, 4))
        self.assertLess(calibration["case_count"], app.MIN_CALIBRATION_CASES)


if __name__ == "__main__":
    unittest.main()
