"""
Analysis Context for MDChat.

Holds per-conversation state: loaded Universe, file paths, computed results,
and generated artifacts. Acts as a shared blackboard between skills.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional


class AnalysisContext:
    """Mutable state container shared across skills within a chat session."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self._store: Dict[str, Any] = {}
        self._artifacts: Dict[str, str] = {}  # name -> file path
        self._history: List[str] = []  # ordered list of skill names executed
        self.output_dir = output_dir or os.path.join(
            tempfile.gettempdir(), f"mdchat_{datetime.now():%Y%m%d_%H%M%S}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ---- basic key-value state ----

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def list_keys(self) -> List[str]:
        return list(self._store.keys())

    # ---- convenience properties for common state ----

    @property
    def universe(self):
        """The loaded MDAnalysis Universe, or None."""
        return self._store.get("universe")

    @universe.setter
    def universe(self, u) -> None:
        self._store["universe"] = u

    @property
    def topology_path(self) -> Optional[str]:
        return self._store.get("topology_path")

    @property
    def trajectory_path(self) -> Optional[str]:
        return self._store.get("trajectory_path")

    # ---- artifacts (generated files) ----

    def add_artifact(self, name: str, path: str) -> None:
        self._artifacts[name] = path

    def get_artifacts(self) -> Dict[str, str]:
        return dict(self._artifacts)

    # ---- execution history ----

    def record_execution(self, skill_name: str) -> None:
        self._history.append(skill_name)

    def get_history(self) -> List[str]:
        return list(self._history)

    # ---- state summary for the LLM system prompt ----

    def get_state_summary(self) -> str:
        """Produce a concise summary of the current analysis state."""
        lines: List[str] = []

        if self.universe is not None:
            u = self.universe
            lines.append(
                f"Trajectory loaded: {self.topology_path or '?'} / "
                f"{self.trajectory_path or '?'}  "
                f"({u.trajectory.n_frames} frames, {u.atoms.n_atoms} atoms)"
            )
        else:
            lines.append("No trajectory loaded yet.")

        data_keys = [k for k in self._store if k not in {
            "universe", "topology_path", "trajectory_path",
        }]
        if data_keys:
            lines.append(f"Computed data available: {', '.join(data_keys)}")

        if self._artifacts:
            lines.append(
                "Artifacts: " + ", ".join(
                    f"{k} ({v})" for k, v in self._artifacts.items()
                )
            )

        if self._history:
            lines.append(f"Skills executed so far: {' -> '.join(self._history)}")

        return "\n".join(lines)
