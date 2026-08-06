"""Helpers for resolving command output directories.

Commands used to write to a bare ``results/<step>/<timestamp>`` path, which
silently depended on the process being launched from the project root. These
helpers make that behavior explicit instead: the results root is configurable
via the ``KCLUSTER_RESULTS_DIR`` environment variable (default ``results``),
and every command resolves its output directory to an absolute path so the
caller can see exactly where files are written regardless of the CWD.
"""

import os
import time

RESULTS_DIR_ENV = "KCLUSTER_RESULTS_DIR"


def results_root() -> str:
    """Base directory for generated results.

    Defaults to ``results`` (relative to the CWD, as before), but can be pinned
    to a fixed absolute location via the ``KCLUSTER_RESULTS_DIR`` env variable so
    an installed ``kcluster`` writes to the same place from any directory.
    """
    return os.environ.get(RESULTS_DIR_ENV, "results")


def timestamp() -> str:
    """A sortable, filesystem-safe timestamp, e.g. ``20260806-142530``."""
    return time.strftime("%Y%m%d-%H%M%S")


def default_output_dir(step: str) -> str:
    """Default output directory for a pipeline ``step`` (e.g. ``concept``)."""
    return os.path.join(results_root(), step, timestamp())


def prepare_output_dir(output_dir: str, *, exist_ok: bool = True) -> str:
    """Resolve ``output_dir`` to an absolute path and create it.

    Returns the absolute path so the command can report where it wrote results.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=exist_ok)
    return output_dir
