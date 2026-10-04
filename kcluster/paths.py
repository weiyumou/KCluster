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
      kc/{concept,kcluster,embed}/  final KC models, dataset-prefixed, by origin
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

import glob
import json
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


def required_result_dir(explicit: str | None = None, explicit_run_dir: str | None = None) -> str:
    """The result dir a command reads from, as an absolute path, or raise.

    Resolution order: the ``--result_dir`` flag, then the run folder. Unlike
    :func:`default_result_dir` nothing is minted: a command that needs an
    earlier step's output has nothing to find in a fresh folder.
    """
    result_dir = explicit or run_dir(explicit_run_dir)
    if not result_dir:
        raise SystemExit("--result_dir is required unless --run_dir (or KCLUSTER_RUN_DIR) is set")
    return os.path.abspath(result_dir)


def dataset_name(data_path: str) -> str:
    """The ``<ds>`` a question file's artifacts are prefixed with: its stem, spaces hyphenated."""
    return os.path.splitext(os.path.basename(data_path))[0].replace(" ", "-")


#: The three origins a KC model can have, each its own subfolder of ``kc/`` (D15).
KC_KINDS = ("concept", "kcluster", "embed")


def kc_dir(result_dir: str, kind: str | None = None) -> str:
    """Final KC models (``kc/<kind>/<ds>_<model>-kc.csv``), organized by origin.

    ``kind`` names the family a writer produces: ``concept`` (the concept
    step, plus concept-cosine — it clusters the concept phrases), ``kcluster``
    (congruity clustering), or ``embed`` (the question-embedding cosine
    models). Omit it for the root, which is what readers scan — recursively,
    so pre-D15 flat dirs keep resolving.
    """
    if kind is None:
        return os.path.join(result_dir, "kc")
    assert kind in KC_KINDS, f"unknown KC kind {kind!r}"
    return os.path.join(result_dir, "kc", kind)


def concept_kc_path(result_dir: str) -> str:
    """The result dir's single Concept KC (``<ds>_concept-kc.csv``), or raise.

    Searched recursively under ``kc/``, so the D15 layout (``kc/concept/``)
    and pre-D15 flat result dirs both resolve.
    """
    kc_root = kc_dir(result_dir)
    match = glob.glob(os.path.join(kc_root, "**", "*_concept-kc.csv"), recursive=True)
    if len(match) != 1:
        raise SystemExit(f"Expected exactly one *_concept-kc.csv under {kc_root}, found {len(match)} — "
                         "run the concept step (or vertex-build-kc) into this result dir first")
    return match[0]


def result_dataset(result_dir: str) -> str:
    """The ``<ds>`` of a result dir, read off its Concept KC's filename."""
    return os.path.basename(concept_kc_path(result_dir)).removesuffix("_concept-kc.csv")


def save_args(result_dir: str, step: str, ds: str, values: dict) -> None:
    """Write a step's provenance breadcrumb (``args-<step>-<ds>.json``) at the result root."""
    with open(os.path.join(result_dir, f"args-{step}-{ds}.json"), "w") as f:
        json.dump(values, f, indent=2)


def _recorded_data_path(result_dir: str, ds: str) -> str | None:
    for fname in sorted(glob.glob(f"args-*-{ds}.json", root_dir=result_dir)):
        with open(os.path.join(result_dir, fname)) as f:
            if recorded := json.load(f).get("data_path"):
                return recorded
    return None


def question_path(result_dir: str, ds: str, explicit: str | None = None) -> str:
    """The question file behind a result dir: ``--data_path``, else the recorded one.

    Any step's breadcrumb records it — ``args-concept`` in a local run,
    ``args-kc`` in a Vertex course dir — so the first that does is used. The
    flag overrides it because a result dir is often read on a different machine
    than it was written on: a cluster run pulled back to a laptop, or a Vertex
    run launched from one, records a path that does not exist where it is now.
    """
    path = explicit or _recorded_data_path(result_dir, ds)
    if not path:
        raise SystemExit(f"No data_path recorded in args-*-{ds}.json under {result_dir} — "
                         "pass --data_path with the question file")
    if not os.path.isfile(path):
        raise SystemExit(f"Question file not found: {path} — pass --data_path with a "
                         "reachable copy of this dataset's questions")
    return path


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


def prepare_output_dir(output_dir: str, *, exist_ok: bool = True) -> str:
    """Resolve ``output_dir`` to an absolute path and create it.

    Returns the absolute path so the command can report where it wrote results.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=exist_ok)
    return output_dir
