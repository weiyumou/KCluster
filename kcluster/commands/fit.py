"""Fit student models to a tagged student-step file and write the comparison.

The last step of the pipeline: `kcluster tag` produces a student-step file
carrying every KC model on identical rows, and this scores them against each
other by fitting a student model (AFM today, PFA alongside it) under each.

Output goes to ``<result dir>/fit/<family>/`` — model-major, so a second family
sits beside the first rather than interleaved with it. One run of the command
can fit several families over one parse of the export (``--model afm --model
pfa``).

    kcluster fit --run_dir results/<run>
    kcluster fit --ss_path <tagged file> --model afm --model pfa --seeds 3
    kcluster fit --run_dir results/<run> -j -1     # fold fits across every core
"""

import argparse
import glob
import os

from kcluster.io.student_step import PREDICTIONS_NAME, TAGGED_SUFFIX
from kcluster.paths import fit_dir, prepare_output_dir, run_dir
from kcluster.tasks.fit import N_FOLDS, N_JOBS, N_SEEDS, SCHEMES, fit_kc_models, kc_models


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

    for family in dict.fromkeys(args.families or ["afm"]):
        outdir = prepare_output_dir(fit_dir(run or os.path.dirname(export), family))
        print(f"*** {family.upper()} on {len(models)} KC model(s) of {export} ***")

        def report(name, row, fit):
            print(f"  {name}: n_params={row['n_params']:,} bic={row['bic']:,.1f}"
                  + "".join(f" {s}={row[f'cv_rmse_{s}']:.4f}" for s in schemes)
                  + ("" if row["is_optimal"] else "  [NOT optimal]"))

        out = fit_kc_models(export, family, models=models, schemes=schemes,
                            n_folds=args.folds, n_seeds=args.seeds,
                            n_jobs=args.jobs, on_model=report)
        _write(out, outdir)
        print(f"*** Wrote {len(out['comparison'])} rows to {outdir} ***")


def _write(out, outdir: str) -> None:
    """The run's tables, in the layout a reader of one dataset's fit expects."""
    out["comparison"].to_csv(os.path.join(outdir, "model-comparison.csv"), index=False)
    out["folds"].to_csv(os.path.join(outdir, "cv-folds.csv"), index=False)
    out["identification"].to_csv(os.path.join(outdir, "identification.csv"), index=False)

    # Only written when the export holds several cohorts, since on one cohort
    # the file would say the same thing about every KC.
    if not out["components"].empty:
        out["components"].to_csv(os.path.join(outdir, "kc-components.csv"), index=False)

    values_dir = prepare_output_dir(os.path.join(outdir, "kc-values"))
    for name, values in out["kc_values"].items():
        values.to_csv(os.path.join(values_dir, f"{name.replace('/', '_')}.csv"), index=False)

    if out["predictions"] is not None:
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
    parser.add_argument("--jobs", "-j", default=N_JOBS, type=int, metavar="N",
                        help=f"Worker processes for the cross-validation fits, -1 for every "
                             f"core (default: {N_JOBS}). Folds are partitioned before any fit "
                             f"starts and collected in order, so this changes the wall clock "
                             f"and not the tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
