"""
Async run logging for parallel MD analysis (stdlib only).

Main process starts a ``QueueListener`` that is the sole writer to
``output/<run_id>/events.jsonl``.  Worker processes attach a
``QueueHandler`` so logging never blocks compute or contends on the file.

Environment variables
---------------------
MD_LOG_LEVEL       Log level (default: INFO)
MD_LOG_ROOT        Parent directory for run folders (default: output)
MD_LOG_QUEUE_SIZE  Bounded queue capacity (default: 10000)
MD_LOG_CONSOLE     1/0 — mirror events to stderr (default: 1)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

_LOGGER_NAME = "md_analysis"
_LOG_RECORD_FIELDS = (
    "run_id",
    "event",
    "component",
    "message",
    "worker_id",
    "step_id",
    "parent_step_id",
    "context",
    "elapsed_ms",
)

_queue: Optional[mp.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None
_run_id: Optional[str] = None
_run_dir: Optional[Path] = None
_dropped_events: int = 0
_step_local = threading.local()


def _resolve_level() -> int:
    name = os.environ.get("MD_LOG_LEVEL", "INFO").upper()
    return getattr(logging, name, logging.INFO)


def _log_root() -> Path:
    return Path(os.environ.get("MD_LOG_ROOT", "output"))


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
        }
        for key in _LOG_RECORD_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        component = getattr(record, "component", record.name)
        event = getattr(record, "event", "log")
        worker = getattr(record, "worker_id", None)
        prefix = f"[{component}] {event}"
        if worker:
            prefix += f" ({worker})"
        return f"{prefix}: {record.getMessage()}"


def active() -> bool:
    """Return True when a :class:`RunContext` is managing the log queue."""
    return _listener is not None


def logging_enabled() -> bool:
    """True in the main process or in a worker with a ``QueueHandler`` attached."""
    if _listener is not None:
        return True
    logger = logging.getLogger(_LOGGER_NAME)
    return any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers)


def _logging_enabled() -> bool:
    return logging_enabled()


def get_queue() -> Optional[mp.Queue]:
    """Return the multiprocessing log queue, or ``None`` if logging is inactive."""
    return _queue


def run_id() -> Optional[str]:
    return _run_id


def run_dir() -> Optional[Path]:
    return _run_dir


def dropped_events() -> int:
    """Events dropped because the bounded queue was full."""
    return _dropped_events


def worker_id() -> str:
    """Best-effort identifier for the current worker process or thread."""
    try:
        from dask.distributed import get_worker

        worker = get_worker()
        if worker is not None:
            return f"dask_{worker.address}"
    except Exception:
        pass

    try:
        return f"mp_{mp.current_process().name}_{os.getpid()}"
    except Exception:
        pass

    return f"thread_{threading.current_thread().ident}"


def attach_worker(
    log_queue: Optional[mp.Queue],
    run_id_value: Optional[str] = None,
) -> None:
    """Attach a ``QueueHandler`` in a child worker (idempotent)."""
    if log_queue is None:
        return

    global _run_id
    if run_id_value is not None:
        _run_id = run_id_value

    logger = logging.getLogger(_LOGGER_NAME)
    if any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
        return

    logger.handlers.clear()
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logger.setLevel(_resolve_level())
    logger.propagate = False


def _current_step_id() -> Optional[str]:
    stack: Optional[list[str]] = getattr(_step_local, "stack", None)
    if not stack:
        return None
    return stack[-1]


def _parent_step_id() -> Optional[str]:
    stack: Optional[list[str]] = getattr(_step_local, "stack", None)
    if not stack or len(stack) < 2:
        return None
    return stack[-2]


def _push_step(step_id: str) -> None:
    if not hasattr(_step_local, "stack"):
        _step_local.stack = []
    _step_local.stack.append(step_id)


def _pop_step() -> None:
    stack: Optional[list[str]] = getattr(_step_local, "stack", None)
    if stack:
        stack.pop()


def log_event(
    event: str,
    message: str = "",
    *,
    level: int = logging.INFO,
    component: str = "md_analysis",
    context: Optional[Dict[str, Any]] = None,
    worker_id_value: Optional[str] = None,
    step_id: Optional[str] = None,
    parent_step_id: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
    exc_info: Any = False,
) -> None:
    """Emit one structured event.  No-op when logging is not configured."""
    if not _logging_enabled():
        return

    logger = logging.getLogger(_LOGGER_NAME)
    extra: Dict[str, Any] = {
        "event": event,
        "component": component,
        "run_id": _run_id,
        "context": context or {},
        "worker_id": worker_id_value or worker_id(),
        "step_id": step_id or _current_step_id(),
        "parent_step_id": parent_step_id or _parent_step_id(),
    }
    if elapsed_ms is not None:
        extra["elapsed_ms"] = round(elapsed_ms, 3)

    try:
        logger.log(level, message, extra=extra, exc_info=exc_info)
    except Exception:
        global _dropped_events
        _dropped_events += 1


@contextmanager
def step(
    name: str,
    *,
    component: str = "md_analysis",
    context: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """Context manager that logs ``step_start`` / ``step_end`` with timing."""
    step_id = uuid.uuid4().hex[:12]
    parent = _current_step_id()
    step_context = {"step_name": name, **(context or {})}
    _push_step(step_id)
    t0 = time.perf_counter()
    log_event(
        "step_start",
        name,
        component=component,
        context=step_context,
        step_id=step_id,
        parent_step_id=parent,
    )
    try:
        yield step_id
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_event(
            "step_error",
            str(exc),
            level=logging.ERROR,
            component=component,
            context={**step_context, "traceback": traceback.format_exc()},
            step_id=step_id,
            parent_step_id=parent,
            elapsed_ms=elapsed_ms,
            exc_info=True,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_event(
            "step_end",
            name,
            component=component,
            context=step_context,
            step_id=step_id,
            parent_step_id=parent,
            elapsed_ms=elapsed_ms,
        )
    finally:
        _pop_step()


class RunContext:
    """Owns the async log queue and ``output/<run_id>/`` artifacts."""

    def __init__(
        self,
        name: str = "run",
        *,
        output_root: Optional[str | Path] = None,
        run_id_value: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.output_root = Path(output_root) if output_root is not None else _log_root()
        self.run_id = run_id_value or uuid.uuid4().hex[:12]
        self.run_dir = self.output_root / self.run_id
        self.manifest = dict(manifest or {})
        self._queue: Optional[mp.Queue] = None
        self._listener: Optional[logging.handlers.QueueListener] = None

    def __enter__(self) -> "RunContext":
        global _queue, _listener, _run_id, _run_dir, _dropped_events

        _dropped_events = 0
        _run_id = self.run_id
        _run_dir = self.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        maxsize = int(os.environ.get("MD_LOG_QUEUE_SIZE", "10000"))
        self._queue = mp.Queue(maxsize=maxsize)
        _queue = self._queue

        events_path = self.run_dir / "events.jsonl"
        file_handler = logging.FileHandler(events_path, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())

        handlers: list[logging.Handler] = [file_handler]
        if os.environ.get("MD_LOG_CONSOLE", "1") not in ("0", "false", "False"):
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(_ConsoleFormatter())
            handlers.append(console_handler)

        self._listener = logging.handlers.QueueListener(
            self._queue,
            *handlers,
            respect_handler_level=True,
        )
        self._listener.start()
        _listener = self._listener

        logger = logging.getLogger(_LOGGER_NAME)
        logger.handlers.clear()
        logger.addHandler(logging.handlers.QueueHandler(self._queue))
        logger.setLevel(_resolve_level())
        logger.propagate = False

        manifest = {
            "run_id": self.run_id,
            "name": self.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "events_path": str(events_path),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
            **self.manifest,
        }
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

        log_event(
            "run_start",
            self.name,
            component=self.name,
            context={"manifest_path": str(manifest_path), "run_dir": str(self.run_dir)},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _queue, _listener, _run_id, _run_dir

        if exc_type is not None:
            log_event(
                "run_error",
                str(exc_val),
                level=logging.ERROR,
                component=self.name,
                context={"traceback": traceback.format_exc()},
                exc_info=(exc_type, exc_val, exc_tb),
            )

        log_event(
            "run_end",
            self.name,
            component=self.name,
            context={"dropped_events": _dropped_events},
        )

        if self._listener is not None:
            self._listener.stop()
        self._listener = None
        _listener = None
        self._queue = None
        _queue = None
        _run_id = None
        _run_dir = None

        logger = logging.getLogger(_LOGGER_NAME)
        logger.handlers.clear()

    @property
    def queue(self) -> Optional[mp.Queue]:
        return self._queue

    @classmethod
    def from_namespace(
        cls,
        args: Any,
        *,
        name: str,
        skip: Optional[set[str]] = None,
        output_root: Optional[str | Path] = None,
    ) -> "RunContext":
        """Build a context from an :class:`argparse.Namespace`."""
        skip = skip or set()
        manifest: Dict[str, Any] = {}
        for key, value in vars(args).items():
            if key in skip:
                continue
            try:
                json.dumps(value)
                manifest[key] = value
            except (TypeError, ValueError, OverflowError):
                manifest[key] = repr(value)[:200]
        return cls(name=name, output_root=output_root, manifest=manifest)


def pool_worker_init(
    log_queue: Optional[mp.Queue],
    run_id_value: Optional[str] = None,
) -> None:
    """``multiprocessing.Pool`` initializer — attach worker logging."""
    attach_worker(log_queue, run_id_value)
