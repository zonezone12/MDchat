from __future__ import annotations

from typing import Callable, Dict, List


class MetricRegistry:
    """Registry for metric definitions that can be dynamically selected."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Callable] = {}
        self._default_metrics: List[str] = []

    def register(self, name: str, metric_func: Callable, default: bool = False) -> None:
        """Register a metric function."""
        self._metrics[name] = metric_func
        if default and name not in self._default_metrics:
            self._default_metrics.append(name)

    def unregister(self, name: str) -> None:
        """Unregister a metric."""
        if name in self._metrics:
            del self._metrics[name]
        if name in self._default_metrics:
            self._default_metrics.remove(name)

    def get(self, name: str):
        """Get a metric function by name."""
        return self._metrics.get(name)

    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())

    def get_default_metrics(self) -> List[str]:
        """Get list of default metric names."""
        return self._default_metrics.copy()

    def has_metric(self, name: str) -> bool:
        """Check if a metric is registered."""
        return name in self._metrics


