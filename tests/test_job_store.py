from __future__ import annotations

import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app
from job_store import CheckpointSession, JobStore


LEDGER = {
    "sample_diagnostics": "usable",
    "feature_ledger": [],
    "most_discriminative_features": [],
    "features_that_should_be_discounted": [],
}
REPORT = {
    "summary": "Report.",
    "confidence": "low",
    "no_listed_candidate_probability": 0.1,
    "candidate_evaluations": [
        {"candidate": "Alice Author", "probability": 0.5},
        {"candidate": "Bob Writer", "probability": 0.4},
    ],
    "limitations": [],
}


class JobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.directory.name) / "jobs.sqlite3")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_store_is_owner_only_and_round_trips_jobs(self) -> None:
        self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)
        self.store.save_job("job", "fp", {"status": "completed", "created_at": time.time(), "result": {"a": 1}})
        self.assertEqual(self.store.load_job("job")["result"], {"a": 1})
        self.store.prune(time.time() + 1)
        self.assertIsNone(self.store.load_job("job"))

    def test_checkpoint_session_replays_only_saved_requests(self) -> None:
        first = CheckpointSession(self.store, "fp")
        key = first.call_key("schema", "instructions", "prompt", "model", "high", False)
        self.assertIsNone(first.get(key))
        first.put(key, {"answer": 1})
        second = CheckpointSession(self.store, "fp")
        self.assertEqual(second.available, 1)
        self.assertEqual(second.get(key), {"answer": 1})
        self.assertIsNone(second.get(second.call_key("schema", "instructions", "changed prompt", "model", "high", False)))
        self.assertEqual(second.reused, 1)
        self.store.clear_checkpoints("fp")
        self.assertEqual(CheckpointSession(self.store, "fp").available, 0)

    def _run(self, client: TestClient) -> dict:
        started = client.post(
            "/api/analyze/start",
            data={
                "mode": "attribution",
                "candidates": "Alice Author\nBob Writer",
                "text_input": "The authors should clarify the main estimate and its proof. " * 12,
            },
        ).json()
        for _ in range(200):
            status = client.get(f"/api/analyze/status/{started['job_id']}").json()
            if status["status"] == "completed":
                return status["result"]
            time.sleep(0.02)
        self.fail("The background analysis did not finish.")

    def test_rerun_after_an_interruption_resumes_finished_model_calls(self) -> None:
        calls: list[str] = []
        fail_final = {"enabled": True}

        def provider(instructions, user_input, schema_name, *_args, **_kwargs):
            calls.append(schema_name)
            if schema_name == "observable_feature_ledger":
                return dict(LEDGER)
            if fail_final["enabled"] and user_input.startswith("This is the final adjudication round"):
                raise app.HTTPException(status_code=502, detail="The provider disconnected.")
            return dict(REPORT)

        with (
            patch.object(app, "JOB_STORE", self.store),
            patch.object(app, "CITATION_NETWORK_ENABLED", False),
            patch.object(app, "PUBLIC_CORPUS_ENABLED", False),
            patch.object(app, "ANALYSIS_REVIEW_PASSES", 1),
            patch.object(app, "ADAPTIVE_MAX_TARGETED_ROUNDS", 0),
            patch.object(app, "_call_provider", side_effect=provider),
            TestClient(app.app) as client,
        ):
            interrupted = self._run(client)
            self.assertEqual(interrupted["review_strategy"], "safe timeout fallback")
            self.assertEqual(len(calls), 3)

            fail_final["enabled"] = False
            calls.clear()
            resumed = self._run(client)
            self.assertEqual(resumed["resumed_model_calls"], 2)
            self.assertEqual(len(calls), 1, "Only the unfinished final adjudication should call the model.")

            calls.clear()
            fresh = self._run(client)
            self.assertEqual(fresh["resumed_model_calls"], 0)
            self.assertEqual(len(calls), 3, "A completed run retires its checkpoints.")

    def test_status_reports_a_run_interrupted_by_a_server_restart(self) -> None:
        self.store.save_job("lost", "fp", {"status": "running", "stage": "Independent review 1", "clues": [], "created_at": time.time()})
        with patch.object(app, "JOB_STORE", self.store), TestClient(app.app) as client:
            status = client.get("/api/analyze/status/lost").json()
        self.assertEqual(status["status"], "error")
        self.assertIn("completed model rounds will be reused", status["detail"])


if __name__ == "__main__":
    unittest.main()
