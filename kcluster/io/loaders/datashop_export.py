"""Loader for DataShop student-step exports.

A driver for a DataShop-sourced dataset needs two things out of the export, and
this module is both of them: the set of steps that defines the dataset's
comparison universe, and the export reduced to the minimal student-step
contract (``kcluster.io.student_step``).

**The export defines the universe, not a KC-model template.** A template is a
second export of the same steps and can only drift from the one that actually
has to agree with the questions; :func:`universe_steps` reconstructs what a
template gave — distinct ``Problem Hierarchy`` / ``Problem Name`` /
``Step Name`` rows sorted on those keys — so step lists come out in template
order. Sorting step names alone gives the same *set* in a different order,
which is enough to change a question's ``ds-step-name`` and every artifact
keyed on it.

Reduction is *reduce-and-filter*, never pass-through, even though an export is
already a student-step file:

- rows whose step resolves to no question are **dropped**, which defines the
  comparison universe once and up front. It is why the legacy KC masking
  (``adjust_datashop_kc``, ``match_other_kc``, the unique-step NaN mask) is not
  needed: every KC model is built over the same rows by construction, rather
  than by keeping several masks in step.
- the file is cut to the contract's columns plus the pass-through ``KC (m)``
  models. Their ``Opportunity (m)`` columns are dropped — the KC tagger
  recomputes opportunities so pass-through and generated models are counted the
  same way.
- row order is the export's own, because it is the tie-break DataShop's
  opportunity counts were produced under, and the closest thing to a canonical
  practice order that this file carries.
"""

import pandas as pd

from kcluster.io.student_step import MINIMAL_COLUMNS

#: What makes a DataShop step distinct. ``Step Name`` alone does not: the same
#: name recurs under different problems.
KEY_COLUMNS = ["Problem Hierarchy", "Problem Name", "Step Name"]

#: Contract columns plus the hierarchy, in the order the reduced file keeps.
EXPORT_COLUMNS = ["Anon Student Id", "Problem Hierarchy", "Problem Name", "Step Name",
                  "First Transaction Time", "First Attempt"]


def load_export(path: str, kc_models: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read the columns a driver needs out of a DataShop student-step export.

    Everything is read as text: student ids, step names and timestamps are
    identifiers, and letting pandas infer types on them silently rewrites
    values (``0123`` -> ``123``) that other files still key on.

    :param path: Path to the downloaded export (tab-delimited).
    :param kc_models: Pass-through KC models to carry, by name.
    """
    columns = EXPORT_COLUMNS + [f"KC ({model})" for model in kc_models]
    export = pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False)
    return export[columns]


def universe_steps(export: pd.DataFrame, kc_models: tuple[str, ...]) -> list[str]:
    """DataShop step names every one of ``kc_models`` tags, in KC-template order.

    A step no pass-through model tags is outside the comparison: nothing in the
    file could give it a label, so keeping it would leave rows that only some
    KC models can score.
    """
    steps = export.drop_duplicates(subset=KEY_COLUMNS).sort_values(KEY_COLUMNS, ignore_index=True)
    tagged = pd.Series(True, index=steps.index)
    for model in kc_models:
        tagged &= steps[f"KC ({model})"].str.strip().ne("")
    return list(dict.fromkeys(steps.loc[tagged, "Step Name"]))


def reduce_to_steps(export: pd.DataFrame, steps) -> pd.DataFrame:
    """Drop the export's rows outside ``steps``, keeping its column set and order."""
    reduced = export[export["Step Name"].isin(set(steps))].reset_index(drop=True)
    if missing := [col for col in MINIMAL_COLUMNS if col not in reduced.columns]:
        raise ValueError(f"export is missing required column(s) {missing}")
    return reduced
