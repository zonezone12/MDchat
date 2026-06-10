"""Tests for async stdlib run logging."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path

import pytest

from src.utils.run_log import (
    RunContext,
    active,
    attach_worker,
    log_event,
    logging_enabled,
    pool_worker_init,
    step,
)


def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_run_context_writes_manifest_and_events(tmp_path: Path) -> None:
    with RunContext("test_run", output_root=tmp_path, manifest={"foo": "bar"}) as run:
        assert active()
        assert run.run_dir.is_dir()
        with step("inner", component="test"):
            log_event("custom", "hello", component="test", context={"x": 1})

    assert not active()
    manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "test_run"
    assert manifest["foo"] == "bar"

    events = _read_events(run.run_dir / "events.jsonl")
    event_names = [e["event"] for e in events]
    assert "run_start" in event_names
    assert "step_start" in event_names
    assert "step_end" in event_names
    assert "custom" in event_names
    assert "run_end" in event_names


def _worker_emit(log_queue: mp.Queue, run_id_value: str) -> None:
    pool_worker_init(log_queue, run_id_value)
    assert logging_enabled()
    log_event(
        "worker_ping",
        "from worker",
        component="worker_test",
        context={"ok": True},
    )


def test_worker_events_reach_main_listener(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MD_LOG_CONSOLE", "0")
    ctx = mp.get_context("spawn")

    with RunContext("worker_run", output_root=tmp_path) as run:
        assert run.queue is not None
        proc = ctx.Process(
            target=_worker_emit,
            args=(run.queue, run.run_id),
        )
        proc.start()
        proc.join(timeout=10)
        assert proc.exitcode == 0

    events = _read_events(run.run_dir / "events.jsonl")
    worker_events = [e for e in events if e.get("event") == "worker_ping"]
    assert worker_events
    assert worker_events[0]["run_id"] == run.run_id
    assert worker_events[0]["context"]["ok"] is True


def test_log_event_noop_without_context() -> None:
    assert not logging_enabled()
    log_event("ignored", "should not raise", level=logging.DEBUG)
