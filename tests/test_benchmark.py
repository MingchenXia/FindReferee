from __future__ import annotations

import contextlib
import io
import json
import tempfile
import types
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


    @staticmethod
    def _result(alice: float, bob: float) -> dict:
        return {
            **RESULT,
            "candidate_evaluations": [
                {"candidate": "Alice Author", "probability": alice},
                {"candidate": "Bob Writer", "probability": bob},
            ],
            "no_listed_candidate_probability": round(1 - alice - bob, 6),
        }

    def test_runs_are_labeled_and_compare_pairs_the_shared_cases(self) -> None:
        with patch.object(app, "_perform_analysis", new=AsyncMock(side_effect=[self._result(0.4, 0.5)] * 4)):
            self._main("run", "--repeat", "2", "--label", "before", "--model", "gpt-test")
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in (self.root / "alpha" / "runs").glob("*.json")]
        self.assertEqual({run["_benchmark"]["label"] for run in runs}, {"before"})
        self.assertTrue(all(run["_benchmark"]["model"] == "gpt-test" for run in runs))
        # The candidate version favors Alice more strongly and is slightly noisy between runs.
        after = [self._result(0.7, 0.2), self._result(0.66, 0.24), self._result(0.7, 0.2), self._result(0.66, 0.24)]
        with patch.object(app, "_perform_analysis", new=AsyncMock(side_effect=after)):
            self._main("run", "--repeat", "2", "--label", "after", "--model", "gpt-test")

        before_only = json.loads(self._main("score", "--json", "--label", "before", "--runs", "all"))
        self.assertEqual(before_only["aggregate"]["case_count"], 4)
        report = json.loads(self._main("compare", "--baseline", "before", "--candidate", "after", "--json"))
        self.assertEqual(report["paired_cases"], 2)
        metrics = {row["key"]: row for row in report["metrics"]}
        self.assertAlmostEqual(metrics["top1_accuracy"]["baseline"], 0.5)
        self.assertAlmostEqual(metrics["top1_accuracy"]["candidate"], 0.5)
        self.assertEqual(metrics["top1_accuracy"]["verdict"], "same")
        cases = {row["case"]: row for row in report["cases"]}
        self.assertAlmostEqual(cases["alpha"]["change"], 0.28)
        self.assertEqual(cases["alpha"]["direction"], "improved")
        self.assertEqual(cases["beta"]["direction"], "worsened")
        self.assertAlmostEqual(cases["alpha"]["run_to_run_spread"], 0.04)
        text = self._main("compare", "--baseline", "before", "--candidate", "after")
        self.assertIn("improved in 1, worsened in 1", text)

    def test_small_changes_are_reported_as_run_to_run_spread(self) -> None:
        with patch.object(app, "_perform_analysis", new=AsyncMock(side_effect=[self._result(0.5, 0.4), self._result(0.6, 0.3)] * 2)):
            self._main("run", "--repeat", "2", "--label", "before", "--model", "gpt-test")
        with patch.object(app, "_perform_analysis", new=AsyncMock(side_effect=[self._result(0.52, 0.38), self._result(0.62, 0.28)] * 2)):
            self._main("run", "--repeat", "2", "--label", "after", "--model", "gpt-test")
        report = json.loads(self._main("compare", "--baseline", "before", "--candidate", "after", "--json", "--cases", "alpha"))
        self.assertEqual(report["cases"][0]["direction"], "within run-to-run spread")

    def test_older_app_versions_parse_files_like_the_current_one(self) -> None:
        folder = self.root / "alpha"
        (folder / "bom.txt").write_bytes("\ufeffText with a byte-order mark. ".encode("utf-8") * 3_000)
        legacy = types.SimpleNamespace(SUPPORTED_EXTENSIONS=app.SUPPORTED_EXTENSIONS, _trim=app._trim)
        with patch.object(benchmark, "app", app):
            current = benchmark._document(folder, "bom.txt")
        with patch.object(benchmark, "app", legacy):
            self.assertEqual(benchmark._document(folder, "bom.txt"), current)
        self.assertTrue(current["truncated"])

    def test_a_second_app_version_needs_its_own_process(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            (Path(other) / "app.py").write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit):
                benchmark._load_app(Path(other))
        self.assertIs(benchmark._load_app(Path(app.__file__).parent), app)

    def test_setup_wizard_builds_cases_with_validation_presets(self) -> None:
        reports = self.root / "reports"
        reports.mkdir()
        (reports / "Orbifold referee report.txt").write_text("Report text. " * 50, encoding="utf-8")
        (reports / "notes.txt").write_text("Not a report.", encoding="utf-8")
        (reports / "manuscript draft.txt").write_text("Manuscript text. " * 50, encoding="utf-8")
        target_root = self.root / "cases"
        answers = iter([
            "", "", "", "Nobody", "", "", "arXiv:2101.00001", f"'{reports / 'manuscript draft.txt'}'", "Some Author",
            "n",  # manuscript draft.txt is not a report
            "n",  # notes.txt is not a report
        ])
        with patch("builtins.input", side_effect=lambda _prompt: next(answers)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(benchmark.main(["--root", str(target_root), "setup", "--from", str(reports)]), 0)
        case = json.loads((target_root / "orbifold" / "case.json").read_text(encoding="utf-8"))
        self.assertEqual(case["candidates"], ["Ya Deng", "Charles Favre", "Mingchen Xia"])
        self.assertEqual(case["expected_author"], "Ya Deng")
        self.assertEqual(case["label_status"], "confirmed")
        self.assertEqual(case["context"], "arXiv:2101.00001")
        self.assertEqual(case["underlying_authors"], ["Some Author"])
        self.assertTrue((target_root / "orbifold" / "report.txt").is_file())
        self.assertTrue((target_root / "orbifold" / "manuscript.txt").is_file())
        # Running setup again skips the case that already exists.
        answers = iter(["n", "n"])
        with patch("builtins.input", side_effect=lambda _prompt: next(answers)), contextlib.redirect_stdout(io.StringIO()) as output:
            benchmark.main(["--root", str(target_root), "setup", "--from", str(reports)])
        self.assertIn("already set up as 'orbifold'", output.getvalue())

    def test_dragged_paths_are_unescaped(self) -> None:
        self.assertEqual(benchmark._dropped_path("/Users/me/Test\\ reports/ms.pdf "), Path("/Users/me/Test reports/ms.pdf"))
        self.assertEqual(benchmark._dropped_path("'/Users/me/Test reports/ms.pdf'"), Path("/Users/me/Test reports/ms.pdf"))
        self.assertIsNone(benchmark._dropped_path("  "))

    def test_check_validates_cases_before_calling_the_model(self) -> None:
        def check() -> tuple[int, str]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = benchmark.main(["--root", str(self.root), "check", "--model", "gpt-test"])
            return code, output.getvalue()

        with patch.object(app, "_call_model", return_value={"reply": "OK", "_provider": "codex", "_model": "gpt-test"}) as call:
            self.assertEqual(check()[0], 0)
        self.assertEqual(call.call_args.args[5], "low")
        with patch.object(app, "_call_model", side_effect=app.HTTPException(status_code=502, detail="Not signed in")):
            code, text = check()
        self.assertEqual(code, 1)
        self.assertIn("Not signed in", text)
        case = json.loads((self.root / "beta" / "case.json").read_text(encoding="utf-8"))
        (self.root / "beta" / "case.json").write_text(json.dumps({**case, "expected_author": "Nobody"}), encoding="utf-8")
        with patch.object(app, "_call_model") as call:
            code, text = check()
        self.assertEqual(code, 1)
        self.assertIn("'Nobody' is not one of the candidates", text)
        call.assert_not_called()

    def test_resume_skips_finished_runs(self) -> None:
        with patch.object(app, "_perform_analysis", new=AsyncMock(return_value=dict(RESULT))) as analysis:
            self._main("run", "--repeat", "2", "--resume", "--label", "v1", "--cases", "alpha")
            self._main("run", "--repeat", "2", "--resume", "--label", "v1", "--cases", "alpha")
            self._main("run", "--repeat", "3", "--resume", "--label", "v1", "--cases", "alpha")
        self.assertEqual(analysis.await_count, 3)
        self.assertEqual(len(list((self.root / "alpha" / "runs").glob("*.json"))), 3)

    def test_provider_failure_stops_the_batch_without_saving_a_run(self) -> None:
        failure = app.HTTPException(status_code=502, detail="No active ChatGPT/Codex subscription was detected.")
        with patch.object(app, "_perform_analysis", new=AsyncMock(side_effect=failure)):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.main(["--root", str(self.root), "run", "--label", "v1"]), 2)
        self.assertFalse(list(self.root.glob("*/runs/*.json")))
        timeout = app.HTTPException(status_code=502, detail="The analysis time budget was exhausted.")
        with patch.object(app, "_perform_analysis", new=AsyncMock(side_effect=timeout)):
            self._main("run", "--label", "v1", "--cases", "alpha")
        saved = json.loads(next((self.root / "alpha" / "runs").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["review_strategy"], "safe timeout fallback")

if __name__ == "__main__":
    unittest.main()
