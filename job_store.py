"""Opt-in local persistence for background analyses and their model checkpoints.

Results and model responses derive from private documents, so nothing is written
unless AUTHOR_ATTRIBUTION_JOB_STORE names a SQLite file. The file is created with
owner-only permissions and rows expire with the job TTL.

Checkpoints make an interrupted analysis resumable: every model response is saved
under a fingerprint of the run's inputs and a hash of the exact request. Starting
the same analysis again replays saved responses for requests that are unchanged,
so only the unfinished rounds call the model. A normally completed run deletes its
checkpoints, so a deliberate rerun still gets fresh, independent answers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    """Stable SHA-256 of a JSON-serializable value."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class JobStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch(mode=0o600)
        os.chmod(self.path, 0o600)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    fingerprint TEXT NOT NULL,
                    call_key TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (fingerprint, call_key)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per operation keeps worker threads independent.
        return sqlite3.connect(self.path, timeout=30)

    def save_job(self, job_id: str, input_fingerprint: str, job: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs (job_id, fingerprint, payload, created_at) VALUES (?, ?, ?, ?)",
                (job_id, input_fingerprint, json.dumps(job, ensure_ascii=False, default=str), float(job.get("created_at", time.time()))),
            )

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def checkpoints(self, input_fingerprint: str) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT call_key, response FROM checkpoints WHERE fingerprint = ?", (input_fingerprint,)
            ).fetchall()
        return {key: json.loads(response) for key, response in rows}

    def save_checkpoint(self, input_fingerprint: str, call_key: str, response: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO checkpoints (fingerprint, call_key, response, created_at) VALUES (?, ?, ?, ?)",
                (input_fingerprint, call_key, json.dumps(response, ensure_ascii=False), time.time()),
            )

    def clear_checkpoints(self, input_fingerprint: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM checkpoints WHERE fingerprint = ?", (input_fingerprint,))

    def prune(self, cutoff: float) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
            connection.execute("DELETE FROM checkpoints WHERE created_at < ?", (cutoff,))


class CheckpointSession:
    """Replays saved model responses for one input fingerprint and records new ones."""

    def __init__(self, store: JobStore, input_fingerprint: str) -> None:
        self.store = store
        self.fingerprint = input_fingerprint
        self._saved = store.checkpoints(input_fingerprint)
        self.available = len(self._saved)
        self.reused = 0

    @staticmethod
    def call_key(*parts: Any) -> str:
        return fingerprint(parts)

    def get(self, call_key: str) -> dict[str, Any] | None:
        saved = self._saved.get(call_key)
        if saved is None:
            return None
        self.reused += 1
        return copy.deepcopy(saved)

    def put(self, call_key: str, response: dict[str, Any]) -> None:
        self._saved[call_key] = copy.deepcopy(response)
        self.store.save_checkpoint(self.fingerprint, call_key, response)
