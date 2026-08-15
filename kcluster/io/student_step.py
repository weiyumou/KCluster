"""Minimal student-step files: the seam between dataset drivers and KC validation.

A *minimal student-step file* is the artifact a dataset driver writes alongside
its question JSONL (``<ds>.jsonl``): the dataset's interaction log reduced to one
first attempt per student-step encounter, carrying no generated KC models. The KC
tagger joins KC models onto it and computes their opportunity counts; the result
is a DataShop-style student-step file that AFM/PFA packages (and LearnSphere)
consume as-is. Only DataShop columns appear, so a minimal file is a valid — if
sparse — student-step export.

The two files are named for the stage they are at, so which one a path holds is
never a guess: ``<ds>_student-step-minimal.txt`` (:data:`MINIMAL_SUFFIX`) and
``<ds>_student-step-tagged.txt`` (:data:`TAGGED_SUFFIX`), both under the same
``<ds>`` as the question JSONL. Where they live follows from what they depend on:
the minimal file is common to every run of the dataset, so it is a working file
in its ``interim/``; a tagged file is one run's KC models joined onto it, so it
lands in that run's result dir — see ``datasets/README.md``.

Schema (tab-delimited, UTF-8, one header row):

==========================  ========  ==================================================
column                      required  contents
==========================  ========  ==================================================
``Anon Student Id``         yes       opaque student id; never PII
``Problem Name``            yes       native DataShop value; sources without a problem
                                      grouping copy ``Step Name``
``Step Name``               yes       native DataShop value; non-DataShop sources use
                                      the question id. Whatever it holds, each
                                      question declares it in ``ds-step-name``
``First Attempt``           yes       ``correct`` / ``incorrect`` / ``hint`` /
                                      ``unknown`` — map the source's outcomes in the
                                      driver, deliberately
``First Transaction Time``  yes       ISO-8601; string sort must equal time sort (below)
``Problem Hierarchy``       no        pass through where the source has one
``KC (<name>)``             no        pass-through *expert* models only, ``~~``-joined,
                                      empty where a step has none
``Opportunity (<name>)``    never     opportunity counting belongs to the tagger, which
                                      computes it for expert and generated models alike
==========================  ========  ==================================================

Three invariants earn their strictness:

- **Every row must resolve to exactly one question** in the accompanying JSONL
  (:func:`check_coverage`). AFM implementations skip rows whose KC cell is
  empty, so a row that no generated KC model can label changes the observation
  count under every generated model while a pass-through expert model keeps it
  — and AIC/BIC across KC models silently stop being comparable, which is the
  comparison this file exists to serve. Resolution is keyed on the questions'
  ``ds-*`` passthrough fields (``~``-joined values expanded, as in
  ``create_datashop_kc``), of which ``ds-step-name`` is **required**: the step a
  question maps to is declared by the driver that wrote both files, never
  inferred here from a naming convention that nothing enforces. A DataShop-derived
  question carries its native value; any other source copies its own id into the
  field and writes the same value to ``Step Name``. The reverse direction — a
  question with no rows — is a per-dataset modelling choice, so it is returned
  for the driver to act on, never raised.

- **String order of ``First Transaction Time`` must agree with time order**
  (:func:`validate_student_step`). Consumers sort this column as text, and that
  ordering is the single definition of "before" shared by opportunity counts
  and every history-dependent feature; a column that sorts differently as text
  than as time silently rebases them all. Constant precision and a single
  timezone offset satisfy this; the check verifies the property, not a format.

- **Expert KC columns must tag exactly the same rows**
  (:func:`validate_student_step`). Comparability is a property of the file as a
  whole: every KC model in a tagged file — expert, generated, and the
  Single-KC/Unique-step defaults the tagger always adds — covers one shared row
  set, which the tagger takes to be the expert-tagged rows. Expert models that
  disagree with each other leave that universe ill-defined, so they must be
  reconciled in the driver, where the data can be examined. Rows tagged by no
  model are kept, not dropped: they are inert in every fit (skipped, and never
  counted toward any opportunity), and they preserve the student's full history.
"""

import itertools
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from kcluster.core.question import Question

MINIMAL_COLUMNS = ("Anon Student Id", "Problem Name", "Step Name",
                   "First Attempt", "First Transaction Time")

#: Filename of the untagged file a driver writes, after its ``<ds>``. It lives in
#: the dataset's ``interim/``: a regenerable input to the tagger, one per dataset.
MINIMAL_SUFFIX = "_student-step-minimal.txt"

#: Filename of the tagged file the tagger writes, after the same ``<ds>``. It
#: lives in the result dir whose KC models it carries: one per run, not per
#: dataset, and what an AFM/PFA package is handed.
TAGGED_SUFFIX = "_student-step-tagged.txt"

#: The DataShop vocabulary for ``First Attempt``; anything else must be mapped
#: by the driver, not passed through.
FIRST_ATTEMPT_VALUES = frozenset({"correct", "incorrect", "hint", "unknown"})

KC_COLUMN = re.compile(r"^KC \((?P<name>.+)\)$")
OPPORTUNITY_COLUMN = re.compile(r"^Opportunity \(.+\)$")

#: Separator for a step with multiple KCs; ``Opportunity`` cells join their
#: counts with the same separator, aligned by position.
MULTI_KC_SEP = "~~"

#: Question passthrough field -> student-step column. The coverage key uses
#: whichever of these the questions carry and the student-step frame has a
#: column for. ``ds-step-name`` is required (see :data:`REQUIRED_KEY_FIELD`);
#: the other two refine the key where a dataset has them.
DS_KEY_FIELDS = {
    "ds-problem-hierarchy": "Problem Hierarchy",
    "ds-problem-name": "Problem Name",
    "ds-step-name": "Step Name",
}

#: Every question joined to student-step data must declare the step it maps to.
#: DataShop-derived questions carry their native value; other sources put their
#: own id here and write the same value to ``Step Name``.
REQUIRED_KEY_FIELD = "ds-step-name"


def load_student_step(path: str) -> pd.DataFrame:
    """Read a (minimal or tagged) student-step file: every cell a string, empty cells ``""``."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def save_student_step(df: pd.DataFrame, path: str) -> None:
    """Write a student-step DataFrame as a tab-delimited file."""
    df.to_csv(path, sep="\t", index=False)


def validate_student_step(df: pd.DataFrame) -> None:
    """Raise ValueError unless ``df`` satisfies the minimal student-step contract.

    Checks the file-local invariants: required columns present and non-empty, no
    ``Opportunity`` columns, the ``First Attempt`` vocabulary, and the
    string-sort == time-sort property of ``First Transaction Time``. The
    cross-artifact invariant — every row resolves to exactly one question — is
    :func:`check_coverage`, which needs the questions.
    """
    if missing := [col for col in MINIMAL_COLUMNS if col not in df.columns]:
        raise ValueError(f"missing required column(s) {missing}")

    if opp := [col for col in df.columns if OPPORTUNITY_COLUMN.match(col)]:
        raise ValueError(f"column(s) {opp} are forbidden in a minimal student-step file: "
                         "the tagger owns opportunity counting, for expert and generated KC models alike")

    for col in MINIMAL_COLUMNS:
        empty = df[col].isna() | df[col].astype(str).str.strip().eq("")
        if empty.any():
            raise ValueError(f"column {col!r} has {int(empty.sum())} empty cell(s)")

    if bad := set(df["First Attempt"]) - FIRST_ATTEMPT_VALUES:
        raise ValueError(f"unrecognized 'First Attempt' value(s) {sorted(bad)}; the vocabulary is "
                         f"{sorted(FIRST_ATTEMPT_VALUES)} — map the source's outcomes in the driver")

    kc_cols = [col for col in df.columns if KC_COLUMN.match(col)]
    if len(kc_cols) > 1:
        first, *rest = ((col, df[col].isna() | df[col].astype(str).str.strip().eq("")) for col in kc_cols)
        for col, mask in rest:
            if not mask.equals(first[1]):
                raise ValueError(f"expert KC columns {first[0]!r} and {col!r} tag different rows "
                                 f"({int((mask != first[1]).sum())} disagreement(s)). Every KC model in a "
                                 "file must cover the same rows or fits under them are incomparable; "
                                 "reconcile the expert models in the driver.")

    _check_time_order(df["First Transaction Time"])


def _check_time_order(times: pd.Series) -> None:
    """Raise ValueError unless sorting ``times`` as text equals sorting them as time."""
    unique = sorted(set(times.astype(str)))
    try:
        parsed = pd.to_datetime(unique, utc=True)
    except (ValueError, TypeError):
        try:  # heterogeneous formats defeat the fast path; parse element-wise
            parsed = pd.to_datetime(unique, format="mixed", utc=True)
        except (ValueError, TypeError) as e:
            raise ValueError(f"unparseable 'First Transaction Time' value: {e}") from e

    backwards = np.diff(parsed.asi8) < 0
    if backwards.any():
        i = int(np.argmax(backwards))
        raise ValueError("'First Transaction Time' sorts differently as text than as time: "
                         f"{unique[i]!r} < {unique[i + 1]!r} as strings but not chronologically. "
                         "Consumers order practice histories by sorting this column as text, so "
                         "rewrite the timestamps with constant precision and a single timezone.")


def _key_values(value) -> list[str]:
    """The values of one passthrough field, in either shape a question comes in.

    A question spanning several DataShop steps holds them as a list in its
    JSONL, and ``~``-joined in a KC CSV — ``Question.flat_dict`` joins them on
    the way out. Both reach here: drivers check their questions as loaded, the
    tagger checks the flattened rows of each KC model.
    """
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value).split("~")


def check_coverage(questions: Sequence[Question], df: pd.DataFrame) -> list[str]:
    """Verify every student-step row resolves to exactly one question.

    Keying: every question must carry :data:`REQUIRED_KEY_FIELD`, matched
    against ``Step Name``; ``ds-problem-hierarchy`` and ``ds-problem-name``
    refine the key where the questions carry them and ``df`` has the column. A
    question covering several steps holds them as a list or ``~``-joined
    (:func:`_key_values`); either way each is expanded into its own key, as
    ``create_datashop_kc`` does.

    :param questions: the dataset's questions, as loaded from ``<ds>.jsonl``
    :param df: a minimal student-step DataFrame
    :raises ValueError: a question lacks the key field, or a row matches no
        question, or a row matches more than one
    :return: ids of questions with no interaction rows — whether to keep or drop
        those questions is the driver's decision, so they are reported, not raised
    """
    if unkeyed := [str(q.get("id", "<no id>")) for q in questions if REQUIRED_KEY_FIELD not in q]:
        raise ValueError(f"{len(unkeyed)} of {len(questions)} questions lack {REQUIRED_KEY_FIELD!r}, "
                         f"e.g. {unkeyed[:5]}. Every question joined to student-step data must declare "
                         "the step it maps to: DataShop-derived questions carry their native value, other "
                         "sources copy their own id into it and write the same value to 'Step Name'.")

    fields = []
    for field in DS_KEY_FIELDS:
        n = sum(field in q for q in questions)
        if 0 < n < len(questions):
            raise ValueError(f"passthrough field {field!r} is present on {n} of {len(questions)} questions; "
                             "it must be on all or none for the step-to-question key to be well-defined")
        if n and DS_KEY_FIELDS[field] in df.columns:
            fields.append(field)

    key_cols = [DS_KEY_FIELDS[field] for field in fields]

    def keys_of(q: Question):
        return itertools.product(*(_key_values(q[field]) for field in fields))

    key_to_ids: dict[tuple, set] = {}
    for q in questions:
        for key in keys_of(q):
            key_to_ids.setdefault(tuple(key), set()).add(str(q["id"]))

    unmatched, ambiguous, covered = [], [], set()
    for key in df[key_cols].astype(str).drop_duplicates().itertuples(index=False, name=None):
        ids = key_to_ids.get(key, set())
        if not ids:
            unmatched.append(key)
        elif len(ids) > 1:
            ambiguous.append((key, sorted(ids)))
        else:
            covered.update(ids)

    if unmatched:
        examples = ", ".join(repr(key) for key in unmatched[:5])
        raise ValueError(f"{len(unmatched)} student-step key(s) on {key_cols} match no question, "
                         f"e.g. {examples}. Rows that no KC model can label are skipped by AFM fits, "
                         "which makes fits under different KC models incomparable; fix the driver's "
                         "key population or drop the rows.")
    if ambiguous:
        key, ids = ambiguous[0]
        raise ValueError(f"{len(ambiguous)} student-step key(s) on {key_cols} match more than one "
                         f"question, e.g. {key!r} -> {ids}; the step-to-question mapping must be "
                         "many-to-one.")

    return [str(q["id"]) for q in questions if str(q["id"]) not in covered]
