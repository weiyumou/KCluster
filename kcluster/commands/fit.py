"""Fit student models to a tagged student-step file and write the comparison.

The last step of the pipeline: `kcluster tag` produces a student-step file
carrying every KC model on identical rows, and this scores them against each
other by fitting a student model (AFM today, PFA alongside it) under each.

Output goes to ``<result dir>/fit/<family>/`` — model-major, so a second family
sits beside the first rather than interleaved with it. One run of the command
can fit several families over one parse of the export (``--model afm --model
pfa``).

**One result dir per scope.** Where the export bundles independent sub-datasets
(``kcluster.tasks.tag.row_scope`` — a multi-course export is one table holding
several), each is fitted on its own and writes to its own result dir, so the run
looks like ``<run>/<course>/fit/<family>/`` and the tables in it are that
course's. This is the point of the command: a KC model that wins pooled AIC won
it on whichever course brought the most rows, which is not a statement about any
course. Two files at the run root cover the whole export — the comparison of
every KC model in every scope, and the predictions of all of them — because
those are the only two things that are about more than one course.

An export with a single scope is unchanged: the run dir *is* that scope's result
dir, so its tables stay at ``<run>/fit/<family>/`` and no root files are written.

    kcluster fit --run_dir results/<run>
    kcluster fit --ss_path <tagged file> --model afm --model pfa --seeds 3
    kcluster fit --run_dir results/<run> -j -1     # fold fits across every core
    kcluster fit --run_dir results/<run> --no_scope   # one pooled fit, as before
"""

import argparse
import contextlib
import glob
import os
import re

import pandas as pd

from kcluster.io.student_step import (
    PREDICTIONS_NAME,
    TAGGED_SUFFIX,
    load_student_step,
)
from kcluster.paths import fit_dir, prepare_output_dir, run_dir
from kcluster.tasks.fit import N_FOLDS, N_JOBS, N_SEEDS, SCHEMES, fit_kc_models, kc_models
from kcluster.tasks.tag import SCOPE_COLUMN, row_scope


def main(args):
    run = run_dir(args.run_dir)
    export = args.ss_path or _default_export(run)
    schemes = args.schemes if args.schemes is not None else list(SCHEMES)

    available = kc_models(export)
    if not available:
        raise SystemExit(f"No 'KC (...)' columns in {export} — is it a tagged file?")
    models = args.kc_models or available
    if unknown := [m for m in models if m not in available]:
        raise SystemExit(f"Unknown KC model(s) {unknown}. The file carries: {available}")

    ss = load_student_step(export)
    root = run or os.path.dirname(os.path.abspath(export))
    scope = None if args.no_scope else row_scope(ss, args.scope_column)
    parts = _scope_parts(ss, scope, root)
    if len(parts) > 1:
        _report_split(ss, scope, models)

    for family in dict.fromkeys(args.families or ["afm"]):
        print(f"*** {family.upper()} on {len(models)} KC model(s) of {export}"
              + (f", {len(parts)} scope(s)" if len(parts) > 1 else "") + " ***")
        comparisons, predictions = [], []

        for name, (result_dir, rows) in parts.items():
            outdir = prepare_output_dir(fit_dir(result_dir, family))
            label = "" if name is None else f"[{name}] "

            def report(model, row, fit, label=label):
                print(f"  {label}{model}: n_params={row['n_params']:,} bic={row['bic']:,.1f}"
                      + ("" if row["is_optimal"] else "  [NOT optimal]"))

            out = fit_kc_models(ss.loc[rows], family, models=models, schemes=schemes,
                                n_folds=args.folds, n_seeds=args.seeds, n_jobs=args.jobs,
                                paired=not args.unpaired, on_model=report)
            _report_cv(out, schemes, label)
            # Predictions are one 200 MB-scale table per family; a scoped run
            # writes the whole of it once at the root instead of a slice of it
            # into each scope, which would store the same rows twice.
            _write(out, outdir, with_predictions=len(parts) == 1)
            print(f"*** Wrote {len(out['comparison'])} rows to {outdir} ***")

            if name is not None:
                comparisons.append(out["comparison"].copy().assign(scope=name))
                predictions.append(out["predictions"])

        if comparisons:
            _write_root(comparisons, predictions, root, export, family)


def _report_cv(out, schemes, label: str) -> None:
    """The cross-validation verdict, once the folds are in.

    Under pairing this cannot be reported per model as it is fitted — no model
    has a score until every design has been scored on the shared partitions — so
    it lands here, as the one line per scheme that a reader is actually after.
    """
    comparison, contrasts = out["comparison"], out["contrasts"]
    for scheme in schemes:
        best = comparison.loc[comparison[f"cv_rmse_{scheme}"].idxmin()]
        line = (f"  {label}{scheme}: best {best['kc_model']} "
                f"rmse={best[f'cv_rmse_{scheme}']:.4f} (sd {best[f'cv_rmse_sd_{scheme}']:.4f})")
        if not contrasts.empty:
            won = contrasts[(contrasts["scheme"] == scheme) & (contrasts["mean_diff"] < 0)]
            line += (f" | {len(won)} of {len(contrasts[contrasts['scheme'] == scheme])} "
                     f"beat {contrasts['baseline'].iloc[0]}")
        print(line)


def _write_root(comparisons, predictions, root: str, export: str, family: str) -> None:
    """The two tables that are about more than one scope, at the run root.

    Named for the dataset and family because nothing in their path is: they sit
    beside the scopes' result dirs rather than inside one, which is the whole
    reason they exist. Everything else a fit produces belongs to one scope and
    stays in that scope's dir.
    """
    stem = os.path.basename(export).removesuffix(TAGGED_SUFFIX)
    prefix = os.path.join(os.path.abspath(root), f"{stem}_{family}-")

    by_scope = pd.concat(comparisons, ignore_index=True)
    by_scope.insert(0, "scope", by_scope.pop("scope"))
    path = f"{prefix}model-comparison-by-scope.csv"
    by_scope.to_csv(path, index=False)
    print(f"*** Wrote {len(by_scope)} rows to {path} ***")

    # sort_index puts the scopes' rows back in the export's own order: the
    # parts were sliced from one frame and each kept its labels.
    if all(part is not None for part in predictions):
        merged = pd.concat(predictions).sort_index()
        path = f"{prefix}{PREDICTIONS_NAME}"
        merged.to_csv(path, sep="\t", index=False, float_format="%.6f", lineterminator="\n")
        print(f"*** Wrote {len(merged):,} rows to {path} ***")


def _slug(value: str) -> str:
    """``value`` as one filesystem-safe path component."""
    return re.sub(r"[^\w.-]+", "-", value).strip("-") or "scope"


def _scope_parts(ss: pd.DataFrame, scope, root: str) -> dict:
    """``{scope name: (result dir, row labels)}`` — what to fit, and where it goes.

    ``{None: (root, every row)}`` when the export holds one scope, which is what
    keeps a single-course run writing exactly where it always did: that run's
    result dir *is* its one scope's.

    A scope's dir is the run's existing one where there is one to reuse — the
    per-course result dirs a batch already wrote its ``kc/`` and ``mat/`` into —
    so a fit lands beside the models it scored rather than in a second directory
    naming the same course. DataShop writes a hierarchy level as ``"<type>
    <name>"`` (``"Course EPLA Physics"``) while those dirs are named for the name
    alone, so the leading token is dropped *only* when doing so matches a dir
    that exists; nothing is guessed away from a scope whose dir has yet to be
    made.
    """
    if scope is None:
        return {None: (root, ss.index)}

    existing = ({e for e in os.listdir(root) if os.path.isdir(os.path.join(root, e))}
                if os.path.isdir(root) else set())
    names = {}
    for value in scope.drop_duplicates().sort_values():
        candidates = [_slug(value.split(" ", 1)[-1]), _slug(value)]
        names[value] = next((c for c in candidates if c in existing), candidates[-1])

    if len(set(names.values())) != len(names):
        clashing = sorted(v for v in names.values() if list(names.values()).count(v) > 1)
        raise SystemExit(
            f"Scopes map onto the same result dir name {clashing[0]!r}, so their tables would "
            "overwrite each other. Rename the scopes in the export, or pass --no_scope to fit "
            "them together.")

    print(f"*** {len(names)} scope(s) -> {', '.join(sorted(names.values()))} ***")
    return {value: (os.path.join(root, name), ss.index[scope == value])
            for value, name in names.items()}


def _report_split(ss: pd.DataFrame, scope, models: list[str]) -> None:
    """Say whether fitting the scopes apart is a restriction of fitting them together.

    It is, exactly, when nothing crosses a scope boundary: the design is then
    block diagonal and the joint fit *is* these fits side by side. A student or
    a KC that spans scopes ties them, and the per-scope fits become a different
    (and, for a per-course question, better posed) model than the pooled one —
    worth saying out loud rather than leaving to be inferred from the numbers.
    """
    students = int((scope.groupby(ss["Anon Student Id"]).nunique() > 1).sum())
    crossing = []
    for name in models:
        labels = ss[f"KC ({name})"]
        tagged = labels.ne("")
        if tagged.any() and (scope[tagged].groupby(labels[tagged]).nunique() > 1).any():
            crossing.append(name)
    if not students and not crossing:
        print("*** Scopes are independent: these fits are the pooled fit, split ***")
        return
    print(f"*** {students} student(s) and {len(crossing)} KC model(s) span scopes"
          + (f" ({', '.join(crossing[:3])}{'...' if len(crossing) > 3 else ''})" if crossing else "")
          + "; per-scope fits differ from a pooled one, not just partition it ***")


def _write(out, outdir: str, *, with_predictions: bool = True) -> None:
    """The run's tables, in the layout a reader of one dataset's fit expects."""
    out["comparison"].to_csv(os.path.join(outdir, "model-comparison.csv"), index=False)
    out["folds"].to_csv(os.path.join(outdir, "cv-folds.csv"), index=False)
    out["identification"].to_csv(os.path.join(outdir, "identification.csv"), index=False)

    # Two tables this run may have nothing to say in: a contrast needs a paired
    # run over more than one KC model, and the component map says the same thing
    # about every KC unless the export holds several cohorts. Where there is
    # nothing to write, any file a previous run left is deleted rather than left
    # to be read beside a comparison table it no longer describes.
    for name, table in (("paired-contrasts.csv", out["contrasts"]),
                        ("kc-components.csv", out["components"])):
        path = os.path.join(outdir, name)
        if table.empty:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
        else:
            table.to_csv(path, index=False)

    values_dir = prepare_output_dir(os.path.join(outdir, "kc-values"))
    for name, values in out["kc_values"].items():
        values.to_csv(os.path.join(values_dir, f"{name.replace('/', '_')}.csv"), index=False)

    if with_predictions and out["predictions"] is not None:
        out["predictions"].to_csv(os.path.join(outdir, PREDICTIONS_NAME), sep="\t", index=False,
                                  float_format="%.6f", lineterminator="\n")


def _default_export(run: str | None) -> str:
    """The tagged file of a result dir — the one `kcluster tag` wrote into it."""
    if not run:
        raise SystemExit("Pass --ss_path, or --run_dir for a result dir holding a tagged file")
    found = sorted(glob.glob(os.path.join(run, f"*{TAGGED_SUFFIX}")))
    if len(found) != 1:
        raise SystemExit(
            f"Expected exactly one *{TAGGED_SUFFIX} in {run}, found {len(found)}"
            + (f": {[os.path.basename(p) for p in found]}" if found else "")
            + ". Pass --ss_path to choose.")
    return found[0]


def add_arguments(parser):
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result dir holding the tagged student-step file; also where "
                             "fit/<family>/ is written")
    parser.add_argument("--ss_path", default=None, type=str,
                        help="Tagged student-step file (<ds>_student-step-tagged.txt), if not "
                             "discoverable from --run_dir")
    parser.add_argument("--model", action="append", dest="families", metavar="FAMILY",
                        choices=("afm", "pfa"),
                        help="Student model to fit; repeat for several (default: afm)")
    parser.add_argument("--kc_model", action="append", dest="kc_models", metavar="NAME",
                        help="KC model to score; repeat for several (default: all in the file)")
    parser.add_argument("--scheme", action="append", dest="schemes", metavar="SCHEME",
                        choices=("student_blocked", "item_blocked", "unstratified",
                                 "response_stratified"),
                        help=f"Cross-validation blocking scheme; repeat for several "
                             f"(default: {' '.join(SCHEMES)})")
    parser.add_argument("--folds", default=N_FOLDS, type=int,
                        help=f"Cross-validation folds (default: {N_FOLDS})")
    parser.add_argument("--seeds", default=N_SEEDS, type=int,
                        help=f"Cross-validation repeats, seeds 0..N-1 (default: {N_SEEDS})")
    parser.add_argument("--unpaired", action="store_true",
                        help="Cross-validate each KC model on its own partitions instead of "
                             "scoring them all on shared folds. Same cost, the same per-model "
                             "means and diagnostics, but no contrasts table: the comparison is "
                             "no longer paired, so a small gap cannot be told from fold-to-fold "
                             "noise. Holds one design in memory at a time where the paired run "
                             "holds all of them, which matters on a million-row export.")
    parser.add_argument("--scope_column", default=SCOPE_COLUMN, type=str,
                        help="Column whose coarsest level separates the export's independent "
                             f"sub-datasets (default: {SCOPE_COLUMN!r}). Each is fitted on its own "
                             "into its own result dir. No effect on an export with one scope.")
    parser.add_argument("--no_scope", action="store_true",
                        help="Fit the export as one model, pooling every scope, and write to "
                             "<run>/fit/<family>/ as an unscoped run does.")
    parser.add_argument("--jobs", "-j", default=N_JOBS, type=int, metavar="N",
                        help=f"Worker processes for the cross-validation fits, -1 for every "
                             f"core (default: {N_JOBS}). Folds are partitioned before any fit "
                             f"starts and collected in order, so this changes the wall clock "
                             f"and not the tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
