"""
Built-in MDChat skills.

Importing this package registers all skills with the default registry.
"""

from . import (  # noqa: F401
    trajectory,
    trajectory_observer_pass,
    metrics,
    plateau,
    endpoints,
    plotting,
    scoring,
    visualization,
    trajectory_movie,
    volume,
    guest,
    clustering,
    fileio,
    gsa,
    selection,
    work_log,
    hpc_batch,
)
