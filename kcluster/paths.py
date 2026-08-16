"""Helpers for resolving command output directories.

Commands used to write to a bare ``results/<step>/<timestamp>`` path, which
silently depended on the process being launched from the project root. These
helpers make that behavior explicit instead: the results root is configurable
via the ``KCLUSTER_RESULTS_DIR`` environment variable (default ``results``),
and every command resolves its output directory to an absolute path so the
caller can see exactly where files are written regardless of the CWD.

Layout for the KC pipeline is **artifact-major** (D10): one folder per
dataset (the *result dir*), organized by what the files are rather than by
which step produced them —

    <result_dir>/
      args-<step>-<ds>.json     provenance, one per step
      kc/                       final KC models, dataset-prefixed
      mat/embed/                question-embedding matrices
      mat/pmi/                  assembled congruity matrices
      mat/pmi/raw/              raw score shards (local engine only)

The steps of a pipeline still run at different times (often as separate
cluster jobs), so pass the same ``--run_dir`` (or set ``KCLUSTER_RUN_DIR``)
to make every step target one result dir; downstream steps then find their
inputs automatically. For a Vertex batch the work dir holds one such result
dir per course. Steps outside the KC pipeline (classify, qgen, kc-refine)
keep the older per-step folders via ``default_output_dir``.
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
    """Default output directory for a non-KC pipeline ``step`` (e.g. ``classify``).

    Inside a run folder this is ``<run>/<step>``; otherwise a fresh run folder
    is minted for this invocation alone (``<results>/<timestamp>/<step>``).
    """
    if run := run_dir(explicit_run_dir):
        return os.path.join(run, step)
    return os.path.join(results_root(), timestamp(), step)


def default_result_dir(explicit_run_dir: str | None = None) -> str:
    """The per-dataset result folder: the run folder, or a fresh timestamped one.

    Commands that only *write* (concept, pmi, vertex-launch) may mint a fresh
    folder; commands that read a result dir should require one instead.
    """
    return run_dir(explicit_run_dir) or os.path.join(results_root(), timestamp())


def kc_dir(result_dir: str) -> str:
    """Final KC models (``<ds>_<model>-kc.csv``)."""
    return os.path.join(result_dir, "kc")


def embed_dir(result_dir: str) -> str:
    """Question-embedding matrices (``<ds>_{sbert,llm}-embed.npy``)."""
    return os.path.join(result_dir, "mat", "embed")


def pmi_dir(result_dir: str) -> str:
    """Assembled congruity matrices (``<ds>_pmi-<tag>.npy``)."""
    return os.path.join(result_dir, "mat", "pmi")


def pmi_raw_dir(result_dir: str) -> str:
    """Raw score shards from the local engine (``predictions_*.pt``)."""
    return os.path.join(result_dir, "mat", "pmi", "raw")


def fit_dir(result_dir: str, family: str) -> str:
    """Student-model scores for one model family (``afm``, ``pfa``).

    Model-major, because a family's tables are the *same* tables computed a
    different way: ``fit/afm/`` and ``fit/pfa/`` belong beside each other
    rather than interleaved. Unlike ``kc/`` and ``mat/`` this holds scoring
    output rather than a KCluster artifact.
    """
    return os.path.join(result_dir, "fit", family)


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
