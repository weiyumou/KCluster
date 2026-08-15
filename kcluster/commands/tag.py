"""Tag a minimal student-step file with KC models and opportunity counts.

The last step before student-model fitting: joins KC CSVs (a result dir's
``kc/*-kc.csv``, extra ``--kc_paths``, or both) onto a dataset's minimal
student-step file, adds the ``Single-KC`` / ``Unique-step``
baselines, and computes every model's ``Opportunity`` column — pass-through
expert models included. The output is a DataShop-style student-step file that
AFM/PFA packages consume as-is.

The minimal file is common to every run of a dataset (it lives in the dataset's
``interim/``), while a tagged file is one run's KC models joined onto it — so it
belongs to that run, and lands in the result dir that supplied the models.

A run may hold its KC models one directory deeper: a Vertex work dir gives each
input bank its own result dir, so a dataset split into per-course banks has one
KC file per model *per course*. Those are parts of one model, not rival models,
so files that share a model name are concatenated here, in memory — a dataset's
joint KC table is an intermediate of tagging, never an artifact to keep in sync.
Each part's labels are namespaced by its bank on the way in, keeping the banks'
KC spaces disjoint (``kcluster.tasks.tag.namespace_kc``).
"""

import argparse
import glob
import os

import pandas as pd

from kcluster.io.student_step import (
    KC_COLUMN,
    MINIMAL_SUFFIX,
    TAGGED_SUFFIX,
    load_student_step,
    save_student_step,
)
from kcluster.paths import kc_dir, run_dir
from kcluster.tasks.tag import bank_name, load_kc_csv, model_name, namespace_kc, tag_student_step


def main(args):
    run = run_dir(args.run_dir)
    kc_paths = [os.path.abspath(path) for path in args.kc_paths]
    kc_paths += _run_dir_kc_paths(run) if run else []

    kc_models = load_kc_models(dict.fromkeys(kc_paths))

    ss = load_student_step(args.ss_path)
    tagged = tag_student_step(ss, kc_models)

    output = args.output or _default_output(args.ss_path, run)
    save_student_step(tagged, output)
    n_models = sum(1 for col in tagged.columns if KC_COLUMN.match(col))
    print(f"*** Wrote {len(tagged):,} rows x {n_models} KC models to {output} ***")


def _run_dir_kc_paths(run: str) -> list[str]:
    """The KC files of a result dir: its own ``kc/``, else its result dirs' own.

    One level down is where a Vertex work dir keeps them, one result dir per
    bank. The run's own ``kc/`` wins when it has files, so a dataset that really
    does cluster as a whole is never mixed with per-bank leftovers.
    """
    own = sorted(glob.glob(os.path.join(kc_dir(run), "*-kc.csv")))
    nested = sorted(glob.glob(os.path.join(run, "*", "kc", "*-kc.csv")))
    return [os.path.abspath(path) for path in (own or nested)]


def load_kc_models(kc_paths) -> dict[str, pd.DataFrame]:
    """Load KC CSVs into ``{model name: frame}``, concatenating same-named parts.

    Several files map to one name when a dataset was clustered in banks (one
    result dir per course, say): each covers its own questions, and the model is
    their concatenation. Two files that name the same *questions* are a genuine
    collision — two rival models under one name — and raise instead.
    """
    by_model: dict[str, list[str]] = {}
    for path in kc_paths:
        by_model.setdefault(model_name(path), []).append(path)

    kc_models = {}
    for name, paths in by_model.items():
        for path in paths:
            print(f"*** KC model '{name}' <- {path} ***")
        frames = [load_kc_csv(path) for path in paths]
        kc_models[name] = frames[0] if len(frames) == 1 else _concat_parts(name, paths, frames)
    return kc_models


def _concat_parts(name: str, paths: list[str], frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One model out of its per-bank parts, or raise if they are not parts.

    Each part's labels are namespaced by its bank, so the banks keep disjoint KC
    spaces: clustering ran per bank, so two banks that both name a KC "Newton's
    third law" arrived there separately, and pooling them would fit one set of
    AFM parameters to steps that were never compared.
    """
    frames = [namespace_kc(frame, bank_name(path)) for path, frame in zip(paths, frames, strict=True)]
    combined = pd.concat(frames, ignore_index=True)
    if "id" in combined.columns and (dup := combined["id"].duplicated()).any():
        raise SystemExit(
            f"{len(paths)} KC files map to the model name {name!r} and {int(dup.sum())} question id(s) "
            f"appear in more than one of them, e.g. {combined.loc[dup, 'id'].iloc[0]!r}. Files sharing a "
            "name are taken to be parts of one model over disjoint questions; two models of the same "
            "name over the same questions have to be renamed.")
    print(f"*** KC model '{name}': {len(paths)} parts concatenated -> {len(combined)} questions ***")
    return combined


def _default_output(ss_path: str, run: str | None) -> str:
    """``<ds>_student-step-minimal.txt`` -> ``<ds>_student-step-tagged.txt``.

    The two names are the dataset convention (``kcluster.io.student_step``), so a
    conventionally named input keeps its ``<ds>`` and swaps stage; anything else
    just gains ``-tagged``.
    """
    name = os.path.basename(ss_path)
    if name.endswith(MINIMAL_SUFFIX):
        name = name[:-len(MINIMAL_SUFFIX)] + TAGGED_SUFFIX
    else:
        name = f"{os.path.splitext(name)[0]}-tagged.txt"
    return os.path.join(run or os.path.dirname(os.path.abspath(ss_path)), name)


def add_arguments(parser):
    parser.add_argument("--ss_path", required=True, type=str,
                        help="Path to a minimal student-step file (interim/<ds>_student-step-minimal.txt)")
    parser.add_argument("--kc_paths", nargs="*", default=[], type=str,
                        help="KC CSVs to add (<ds>_<model>-kc.csv; the model becomes the column name)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="A result dir whose kc/*-kc.csv models are all added — or, for a work dir "
                             "of per-bank result dirs, each of their kc/*-kc.csv")
    parser.add_argument("--output", default=None, type=str,
                        help="Output path (default: <ds>_student-step-tagged.txt in --run_dir, which is "
                             "where a tagged file belongs; else beside the input)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
