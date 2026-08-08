"""Helpers for resolving command output directories.

Commands used to write to a bare ``results/<step>/<timestamp>`` path, which
silently depended on the process being launched from the project root. These
helpers make that behavior explicit instead: the results root is configurable
via the ``KCLUSTER_RESULTS_DIR`` environment variable (default ``results``),
and every command resolves its output directory to an absolute path so the
caller can see exactly where files are written regardless of the CWD.

Layout is **run-major**: one folder per run, with a subfolder per step —
``<results>/<run>/{concept,pmi,kc,...}``. The steps of a pipeline run at
different times (often as separate cluster jobs), so pass the same
``--run_dir`` (or set ``KCLUSTER_RUN_DIR``) to keep them together; downstream
steps then find their inputs automatically instead of being handed a
timestamp that has to be paired up by hand.
"""

import os
import time

RESULTS_DIR_ENV = "KCLUSTER_RESULTS_DIR"
RUN_DIR_ENV = "KCLUSTER_RUN_DIR"


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


def run_dir(explicit: str | None = None) -> str | None:
    """The run folder shared by every step of one pipeline run, if any.

    Resolution order: the ``--run_dir`` flag, then ``KCLUSTER_RUN_DIR``.
    """
    return explicit or os.environ.get(RUN_DIR_ENV) or None


def default_output_dir(step: str, explicit_run_dir: str | None = None) -> str:
    """Default output directory for a pipeline ``step`` (e.g. ``concept``).

    Inside a run folder this is ``<run>/<step>``; otherwise a fresh run folder
    is minted for this invocation alone (``<results>/<timestamp>/<step>``).
    """
    if run := run_dir(explicit_run_dir):
        return os.path.join(run, step)
    return os.path.join(results_root(), timestamp(), step)


def step_dir(step: str, explicit_run_dir: str | None = None) -> str | None:
    """Where a previous ``step`` of this run wrote its output, if discoverable.

    Returns None when no run folder is in play, so callers can fall back to
    requiring an explicit path.
    """
    if run := run_dir(explicit_run_dir):
        return os.path.join(run, step)
    return None


def prepare_output_dir(output_dir: str, *, exist_ok: bool = True) -> str:
    """Resolve ``output_dir`` to an absolute path and create it.

    Returns the absolute path so the command can report where it wrote results.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=exist_ok)
    return output_dir
