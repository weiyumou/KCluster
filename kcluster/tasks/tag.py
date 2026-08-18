"""The KC tagger: joins KC models onto a minimal student-step file.

This is the second stage of the dataset seam described in
:mod:`kcluster.io.student_step`: a dataset driver emits ``<ds>.jsonl`` plus a
minimal student-step file, and this module turns the latter into a DataShop-style
student-step file ready for AFM/PFA fitting — KC CSVs from a result dir joined
on as ``KC (<model>)`` columns, the ``Single-KC`` / ``Unique-step`` baselines
added, and an ``Opportunity`` column computed for every KC model, pass-through
expert models included. A KC model whose name collides with a column already in
the file raises, so a DataShop export passing through its own ``Single-KC`` is
caught rather than silently overwritten.

Uniform coverage is enforced, not assumed: the rows the expert KC columns tag
(all rows, if there are none) are the comparable universe. Every joined model
must label all of them and is blanked outside them, so every ``KC`` column in
the output tags exactly the same rows and fits under different models score
the same observations.

Opportunity counts are 1-based and ``~~``-joined for multi-KC steps, under the
canonical ordering DataShop and AFM packages share: each student's rows sorted
by ``First Transaction Time`` *as text*, ties broken by file row order.

Where the file bundles independent sub-datasets — a multi-course export is one
table holding several — the student id and the ``Single-KC`` baseline are scoped
to their group, so every KC model in the output separates over the same groups
and a fit under any of them is the several models the export actually holds
(:func:`row_scope`, :func:`tag_student_step`).
"""

import os

import pandas as pd

from kcluster.io.datashop import create_datashop_kc
from kcluster.io.student_step import (
    DS_KEY_FIELDS,
    KC_COLUMN,
    MULTI_KC_SEP,
    OPPORTUNITY_COLUMN,
    check_coverage,
    validate_student_step,
)

SINGLE_KC_NAME = "Single-KC"
UNIQUE_STEP_NAME = "Unique-step"

#: Separates a bank's name from a KC label when parts of one model are merged
#: (``"EPLA-Physics: Newton's third law"``). Clustering runs per bank, so two
#: banks' labels are alike only by coincidence — a shared spelling would
#: otherwise pool their steps into one KC and one set of AFM parameters.
BANK_KC_SEP = ": "

#: The column whose coarsest level scopes a row: which independent sub-dataset
#: it belongs to. ``Problem Hierarchy`` is DataShop's comma-joined level path
#: (``"Course X"``, ``"sequence S, unit U, module M"``), so its first segment is
#: the coarsest unit the export names — a course in a multi-course export, and
#: the one course itself in a single-course one, where scoping is then a no-op.
SCOPE_COLUMN = "Problem Hierarchy"

#: Separates a scope from the identifier it qualifies (``"Course EPLA Physics:
#: A1B2"``). The same shape and the same reason as :data:`BANK_KC_SEP`, applied
#: to the other side of the design: two courses that share a student id arrived
#: there by enrolment, not by the student being one AFM subject across both, and
#: pooling them fits one intercept to work that was never compared.
SCOPE_SEP = ": "


def row_scope(ss: pd.DataFrame, column: str = SCOPE_COLUMN) -> pd.Series | None:
    """Each row's independent sub-dataset, or ``None`` when the file is one.

    A student-step file can bundle groups that never met the same material —
    courses, sections, cohorts. Fitted together they are not one model but
    several sharing a table: AFM's likelihood separates over them exactly, so
    the pooled log-likelihood, AIC and BIC are sums whose verdict belongs to
    whichever group brought the most rows. Naming the groups here is what lets
    every later stage treat them as the separate models they are.

    The scope is the coarsest level of ``column`` (see :data:`SCOPE_COLUMN`),
    which for a column carrying no level path is the whole value. ``None`` comes
    back when the file has no such column or only one scope in it — both mean
    there is nothing to separate, and callers then leave the file alone.

    A scope has to partition the *material*, not just label it. If one step
    appears under two scopes, those groups share work, their students are not
    independent, and splitting a student across them would fit two intercepts to
    one course's worth of practice — so that raises rather than quietly
    producing a model nobody asked for.
    """
    if column not in ss.columns:
        return None
    scope = ss[column].fillna("").astype(str).str.split(",").str[0].str.strip()
    if scope.nunique() < 2:
        return None
    if scope.eq("").any():
        raise ValueError(
            f"{int(scope.eq('').sum())} row(s) carry no {column!r} value while others do, so the "
            "file names some of its groups and not the rest. An unnamed group would scope to the "
            "empty string and read as one more group. Fill the column, or turn scoping off.")

    per_step = scope.groupby([ss["Problem Name"], ss["Step Name"]]).nunique()
    if len(crossing := per_step.index[per_step > 1]):
        raise ValueError(
            f"{len(crossing)} step(s) appear under more than one {column!r} scope, e.g. "
            f"{tuple(crossing[0])!r}. A scope that does not partition the material would split "
            "students across groups that share work, fitting each of them an intercept from part "
            "of their practice. Scope on a column that does partition it, or turn scoping off to "
            "keep one intercept per student id.")
    return scope


def model_name(kc_path: str) -> str:
    """Model name of a D10 KC file (``<ds>_<model>-kc.csv`` -> ``<model>``)."""
    stem = os.path.splitext(os.path.basename(kc_path))[0]
    return stem.split("_", 1)[-1].removesuffix("-kc")


def bank_name(kc_path: str) -> str:
    """The ``<ds>`` a D10 KC file was built from (``<ds>_<model>-kc.csv`` -> ``<ds>``)."""
    stem = os.path.splitext(os.path.basename(kc_path))[0]
    return stem.split("_", 1)[0]


def namespace_kc(kc: pd.DataFrame, bank: str) -> pd.DataFrame:
    """A copy of ``kc`` with every KC label prefixed by ``bank``.

    Multi-KC cells are prefixed label by label, and an empty label stays empty:
    a model that leaves a question unlabeled must still read as unlabeled to
    :func:`_join_kc_model`, which is what catches an incomplete model.
    """
    def prefixed(cell: str) -> str:
        labels = str(cell).split(MULTI_KC_SEP)
        return MULTI_KC_SEP.join(f"{bank}{BANK_KC_SEP}{label}" if label.strip() else label
                                 for label in labels)

    kc = kc.copy()
    kc["KC"] = kc["KC"].map(prefixed)
    return kc


def load_kc_csv(path: str) -> pd.DataFrame:
    """Read a KC CSV: every cell a string, empty cells ``""`` (as ``load_student_step`` does)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def tag_student_step(ss: pd.DataFrame, kc_models: dict[str, pd.DataFrame], *,
                     scope_column: str | None = SCOPE_COLUMN) -> pd.DataFrame:
    """Tag a minimal student-step frame with KC models and their opportunities.

    Where the file bundles independent sub-datasets (:func:`row_scope`), the two
    things that would otherwise tie them together are scoped to their group: the
    student id, so a student enrolled in two courses is two subjects rather than
    one carrying a single ability across both, and the ``Single-KC`` baseline, so
    "everything here is one skill" is a claim about a course rather than about
    every course at once. Both are prefixed with the scope (:data:`SCOPE_SEP`),
    the same device ``namespace_kc`` already applies to per-bank KC labels, and
    together they leave every KC model in the file separating over exactly the
    groups the export bundles. ``Unique-step`` keys on the hierarchy already and
    generated models arrive bank-namespaced, so neither needs anything here.

    Scoping is what the file already implies rather than a new modelling claim,
    but it is a claim: pass ``scope_column=None`` for a dataset whose groups do
    share subjects — a curriculum sequence a cohort moves through — where one
    intercept per student id is the point.

    :param ss: a minimal student-step frame (``load_student_step``); validated here
    :param kc_models: model name -> KC CSV frame (``load_kc_csv``), joined in order
    :param scope_column: column whose coarsest level separates the sub-datasets,
        or ``None`` to leave students and ``Single-KC`` pooled across the file
    :return: a new frame with one ``KC (<model>)`` + ``Opportunity (<model>)``
        pair per model — the ones joined here, the two default baselines, and
        the expert models the file came with
    """
    validate_student_step(ss)
    ss = ss.reset_index(drop=True).copy()

    scope = row_scope(ss, scope_column) if scope_column else None
    if scope is not None:
        ss["Anon Student Id"] = scope + SCOPE_SEP + ss["Anon Student Id"]

    expert_cols = [col for col in ss.columns if KC_COLUMN.match(col)]
    taken = {KC_COLUMN.match(col)["name"] for col in expert_cols}
    for name in [*kc_models, SINGLE_KC_NAME, UNIQUE_STEP_NAME]:
        if name in taken:
            raise ValueError(f"KC model name {name!r} collides with a KC column already in the file")
        taken.add(name)

    # The comparable universe: the rows the expert models tag (they agree with
    # each other, per validate_student_step), or every row when there are none.
    if expert_cols:
        mask = ss[expert_cols[0]].fillna("").astype(str).str.strip().ne("")
    else:
        mask = pd.Series(True, index=ss.index)

    for name, kc in kc_models.items():
        ss = _join_kc_model(ss, kc, name, mask)

    _add_default_kc(ss, mask, scope)
    return add_opportunities(ss)


def _join_kc_model(ss: pd.DataFrame, kc: pd.DataFrame, name: str, mask: pd.Series) -> pd.DataFrame:
    """Join one KC CSV onto ``ss`` as ``KC (name)``, blanked outside ``mask``."""
    # Every row in the universe must resolve to exactly one question of this
    # model (unmatched, ambiguous, or unkeyed raise); rows outside it are inert
    # and may match nothing. The flattened CSV rows duck-type Question here.
    check_coverage(kc.to_dict("records"), ss.loc[mask])

    fields = [field for field in DS_KEY_FIELDS if field in kc.columns and DS_KEY_FIELDS[field] in ss.columns]
    join_kc = kc[fields + ["KC"]].copy()
    for field in fields:
        # Collapse self-duplicates ("a~a", one question listing a step twice):
        # the explode-join would otherwise duplicate its rows.
        join_kc[field] = join_kc[field].str.split("~").map(lambda parts: "~".join(dict.fromkeys(parts)))

    n_rows = len(ss)
    ss = create_datashop_kc(join_kc, ss, kc_cols=["KC"], new_kc_names=[name])
    assert len(ss) == n_rows, "the step-to-question join duplicated rows"

    col = f"KC ({name})"
    unlabeled = mask & (ss[col].isna() | ss[col].astype(str).str.strip().eq(""))
    if unlabeled.any():
        raise ValueError(f"KC model {name!r} leaves {int(unlabeled.sum())} expert-tagged row(s) without a "
                         "label. Every KC model must cover the same rows or fits under them are incomparable; "
                         "rebuild the model from this dataset's questions.")
    ss[col] = ss[col].where(mask, "")
    return ss


def _add_default_kc(ss: pd.DataFrame, mask: pd.Series, scope: pd.Series | None = None) -> None:
    """Add the two baselines every comparison needs, blanked outside ``mask``.

    Same construction as ``create_default_kc`` (which serves the legacy
    DataShop-template workflow under its own ``-full`` names): a constant KC,
    and one KC per distinct ``Problem Hierarchy + Problem Name + Step Name``
    value. DataShop publishes models of these names built under a
    student-count threshold that drops thin steps; we never apply it, so these
    cover the whole comparable universe.

    The constant is constant *within a scope*, one KC per group rather than one
    for the file. ``Unique-step`` needs no such treatment: it already keys on
    ``Problem Hierarchy``, so its KCs never spanned a group in the first place —
    scoping the constant is what brings the two baselines into line.
    """
    single = (pd.Series(SINGLE_KC_NAME, index=ss.index) if scope is None
              else scope + SCOPE_SEP + SINGLE_KC_NAME)
    ss[f"KC ({SINGLE_KC_NAME})"] = single.where(mask, "")

    parts = [ss[col] for col in ("Problem Hierarchy", "Problem Name", "Step Name") if col in ss.columns]
    steps = parts[0].fillna("").astype(str)
    for part in parts[1:]:
        steps = steps + part.fillna("").astype(str)
    unique_keys = pd.unique(steps[mask])
    key_map = dict(zip(unique_keys, (f"KC-{i}" for i in range(1, len(unique_keys) + 1)), strict=True))
    ss[f"KC ({UNIQUE_STEP_NAME})"] = steps.map(key_map).where(mask, "")


def add_opportunities(ss: pd.DataFrame) -> pd.DataFrame:
    """Give every ``KC (m)`` column an ``Opportunity (m)`` column right after it.

    Counts are 1-based per student and KC; a ``~~``-joined multi-KC cell gets
    ``~~``-joined counts aligned by position, and untagged rows get ``""``.
    Ordering matches what AFM packages recompute: each student's rows by
    ``First Transaction Time`` as text, file row order breaking ties.
    """
    ss = ss.reset_index(drop=True).copy()
    order = ss.sort_values(["Anon Student Id", "First Transaction Time"], kind="stable").index

    for kc_col in [col for col in ss.columns if KC_COLUMN.match(col)]:
        labels = ss[kc_col].fillna("").astype(str).str.split(MULTI_KC_SEP)
        exploded = pd.DataFrame({"student": ss["Anon Student Id"], "kc": labels}).loc[order].explode("kc")
        exploded = exploded[exploded["kc"].ne("")]
        if exploded.reset_index().duplicated(subset=["index", "kc"]).any():
            raise ValueError(f"column {kc_col!r} lists the same KC twice on one step; "
                             "consumers key {kc: opportunity} per step, so the counts would misalign")

        counts = exploded.groupby(["student", "kc"], sort=False).cumcount().add(1).astype(str)
        opp = counts.groupby(level=0, sort=False).agg(MULTI_KC_SEP.join)
        name = KC_COLUMN.match(kc_col)["name"]
        ss[f"Opportunity ({name})"] = opp.reindex(ss.index, fill_value="")

    # DataShop exports pair each model's KC and Opportunity columns; keep that shape.
    columns = []
    for col in ss.columns:
        if OPPORTUNITY_COLUMN.match(col):
            continue
        columns.append(col)
        if kc := KC_COLUMN.match(col):
            columns.append(f"Opportunity ({kc['name']})")
    return ss[columns]
