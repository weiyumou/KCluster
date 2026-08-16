"""Fit student models to a tagged student-step file, one row per KC model.

The step after tagging, and the reason tagging exists: a KC model is judged by
how well a student model fits under it, so this scores every ``KC (<name>)``
column of one tagged export on identical observations and returns the
comparison.

The modelling lives in [Leapfit](https://github.com/weiyumou/LeapFit) and stays
there. What is here is everything that is *not* family-specific — loop the KC
models, cross-validate under each requested blocking scheme, assemble the
tables — so a family is a pair of Leapfit callables (:func:`family_callables`)
rather than a module of its own, matching Leapfit's own decision to make AFM
and PFA siblings behind one interface. BKT joins the dict when it ships.

Leapfit is an optional dependency (the ``fit`` extra), imported lazily so a
base install keeps working; :func:`family_callables` raises with the install
command when it is missing.
"""

import numpy as np
import pandas as pd

# The protocol these tables are produced under (D13). Fixed here rather than
# per call, because the point of the tables is comparison: two runs made under
# different conventions cannot be read side by side.
CONVENTION = "per_fold"      # mean of fold RMSEs, the PyAFM/LearnSphere convention
METHOD = "TNC"               # what LearnSphere used, so fits stay comparable to it
MAX_FUN = 500_000
SCHEMES = ("student_blocked", "item_blocked")
N_FOLDS = 3
N_SEEDS = 10

INSTALL_HINT = (
    "Fitting student models needs Leapfit, which is not part of a base install.\n"
    '  pip install "kcluster[fit]"\n'
    "Leapfit is released on GitHub rather than PyPI, so the extra carries the "
    "release URL; installing 'leapfit' by name will not find it."
)


def family_callables(family: str):
    """``(build_design, fit)`` for a model family, or raise with what to install."""
    try:
        import leapfit
    except ImportError as exc:
        raise SystemExit(INSTALL_HINT) from exc

    families = {
        "afm": (leapfit.build_afm_design, leapfit.fit_afm),
        "pfa": (leapfit.build_pfa_design, leapfit.fit_pfa),
    }
    if family not in families:
        raise ValueError(f"Unknown model family {family!r}; choose from {sorted(families)}")
    return families[family]


def kc_models(export: str) -> list[str]:
    """The KC models a tagged student-step file carries."""
    from leapfit import list_kc_models

    return list_kc_models(export)


def fit_kc_models(export: str, family: str = "afm", *, models=None,
                  schemes=SCHEMES, n_folds: int = N_FOLDS, n_seeds: int = N_SEEDS,
                  on_model=None) -> dict:
    """Fit ``family`` under every KC model of ``export``.

    Returns ``{"comparison", "folds", "identification", "components",
    "kc_values", "predictions"}`` — the comparison table one row per KC model,
    the per-seed cross-validation detail behind its means, every aliased column
    with the reason it was dropped, the KC-to-component map (see
    :func:`_component_rows`), per-KC parameters in DataShop's layout, and the
    export with one predicted-error-rate column per model.

    ``on_model`` is called with ``(name, row, fit)`` after each model, so a
    command can report progress on a run measured in hours.
    """
    from leapfit import load_student_step, repeated_cross_validate

    build_design, fit_model = family_callables(family)
    models = list(models or kc_models(export))

    rows, fold_frames, aliased, components = [], [], [], []
    kc_values, predictions = {}, None

    for name in models:
        data = load_student_step(export, kc_model=name)
        # Identification is the expensive half (a dense rank check), so build
        # once unidentified and identify that: the raw design is what the
        # component map has to be read off anyway, since identification drops
        # a student column per component.
        raw = build_design(data, identify=False)
        design = raw.identify()
        fit = fit_model(design, data.y, method=METHOD, max_fun=MAX_FUN)

        row = {
            "kc_model": name,
            "n_kcs": len(data.kc_names),
            "n_students": len(data.student_names),
            "n_obs": len(data),
            "n_params": design.n_params,
            "n_aliased": len(design.aliased),
            "n_separated": len(fit.separated),
            "log_likelihood": fit.ll,
            "aic": fit.aic,
            "bic": fit.bic,
            "is_optimal": fit.is_optimal,
        }
        for scheme in schemes:
            table = repeated_cross_validate(
                design, data, seeds=list(range(n_seeds)), scheme=scheme,
                n_folds=n_folds, convention=CONVENTION, method=METHOD,
                max_fun=MAX_FUN)
            row |= {
                f"cv_rmse_{scheme}": table["rmse"].mean(),
                f"cv_rmse_sd_{scheme}": table["rmse"].std(ddof=1) if len(table) > 1 else np.nan,
                f"cv_unseen_{scheme}": table["unseen_column_fraction"].mean(),
                f"cv_converged_{scheme}": bool(table["all_converged"].all()),
            }
            table.insert(0, "kc_model", name)
            fold_frames.append(table)
        rows.append(row)

        aliased += [{"kc_model": name, "column": column, "reason": reason}
                    for column, reason in zip(design.aliased.columns,
                                              design.aliased.reasons)]
        components += _component_rows(name, raw, data.kc_names)
        kc_values[name] = fit.kc_values(data)
        predictions = fit.annotate(data, into=predictions)

        if on_model is not None:
            on_model(name, row, fit)

    return {
        "comparison": pd.DataFrame(rows),
        "folds": pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame(),
        "identification": pd.DataFrame(aliased, columns=["kc_model", "column", "reason"]),
        "components": pd.DataFrame(components, columns=["kc_model", "KC Name", "component"]),
        "kc_values": kc_values,
        "predictions": predictions,
    }


def _component_rows(name: str, design, kc_names) -> list[dict]:
    """Each KC's connected component — empty unless the export has several.

    A student-step file can hold groups of students that never met the same
    material: courses, sections, conditions. Leapfit breaks one sum redundancy
    per group, which leaves KC *intercepts* anchored per group — comparable
    within a component, not across, and no recentring can fix that, because
    nothing in the data relates cohorts that share no material. So when there
    is more than one, the mapping is a result of the run rather than a
    footnote. One component (the ordinary case) contributes nothing.
    """
    labels = design.row_components()
    if labels.size == 0 or labels.max() == 0:
        return []
    kc = next(b for b in design.blocks if b.name == "kc_intercept").matrix.tocsc()
    rows = []
    for j, label in enumerate(kc_names):
        lo, hi = kc.indptr[j], kc.indptr[j + 1]
        if lo != hi:  # a KC nobody meets belongs to no cohort
            rows.append({"kc_model": name, "KC Name": label,
                         "component": int(labels[kc.indices[lo]]) + 1})
    return rows
