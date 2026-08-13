"""Tag a minimal student-step file with KC models and opportunity counts.

The last step before student-model fitting: joins KC CSVs (a result dir's
``kc/*-kc.csv``, extra ``--kc_paths``, or both) onto a dataset's minimal
student-step file, adds the ``Single-KC`` / ``Unique-step``
baselines, and computes every model's ``Opportunity`` column — pass-through
expert models included. The output is a DataShop-style student-step file that
AFM/PFA packages consume as-is.
"""

import argparse
import glob
import os

from kcluster.io.student_step import KC_COLUMN, load_student_step, save_student_step
from kcluster.paths import kc_dir, run_dir
from kcluster.tasks.tag import load_kc_csv, model_name, tag_student_step


def main(args):
    kc_paths = [os.path.abspath(path) for path in args.kc_paths]
    if run := run_dir(args.run_dir):
        kc_paths += sorted(os.path.abspath(path) for path in glob.glob(os.path.join(kc_dir(run), "*-kc.csv")))

    kc_models = {}
    for path in dict.fromkeys(kc_paths):
        name = model_name(path)
        if name in kc_models:
            raise SystemExit(f"Two KC files map to the same model name {name!r}; rename one of them")
        print(f"*** KC model '{name}' <- {path} ***")
        kc_models[name] = load_kc_csv(path)

    ss = load_student_step(args.ss_path)
    tagged = tag_student_step(ss, kc_models)

    output = args.output or _default_output(args.ss_path, run_dir(args.run_dir))
    save_student_step(tagged, output)
    n_models = sum(1 for col in tagged.columns if KC_COLUMN.match(col))
    print(f"*** Wrote {len(tagged):,} rows x {n_models} KC models to {output} ***")


def _default_output(ss_path: str, run: str | None) -> str:
    stem = os.path.splitext(os.path.basename(ss_path))[0]
    return os.path.join(run or os.path.dirname(os.path.abspath(ss_path)), f"{stem}-tagged.txt")


def add_arguments(parser):
    parser.add_argument("--ss_path", required=True, type=str,
                        help="Path to a minimal student-step file (<ds>_student-step.txt)")
    parser.add_argument("--kc_paths", nargs="*", default=[], type=str,
                        help="KC CSVs to add (<ds>_<model>-kc.csv; the model becomes the column name)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="A result dir whose kc/*-kc.csv models are all added")
    parser.add_argument("--output", default=None, type=str,
                        help="Output path (default: <input stem>-tagged.txt in --run_dir, else beside the input)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
