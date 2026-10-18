"""Fill a DataShop KC-model export with a run's KC models, for import into DataShop.

DataShop imports KC models from a file shaped like its own KC-model export: one
row per step, matched on ``Step ID``, a ``KC (<model>)`` column per model, and
every other column as exported. This adds a result dir's KC models (and any
``--kc_paths``) to such an export, found and named as ``kcluster tag`` finds and
names them, and joined by the same rules: each step resolves to exactly one
question through the ``ds-*`` keys the questions carry, and every model labels
each step the export's own KC models all tag (every step, if it has none) and
no other. Export the template with the models that should set those steps — an
expert model, or DataShop's ``Unique-step`` to follow its student threshold —
and the imported models are fitted over the same steps as they are.

The export's own models are left out of the output, as DataShop advises:
importing them again would overwrite them. The exception is
``--restrict_suffix``, for a question bank that reaches fewer steps than those
models tag: the new models then label only the bank's steps, and the export's
models go out again under the suffix, cut down to the same steps.
"""

import os

from kcluster.commands.tag import load_kc_models, run_dir_kc_paths
from kcluster.datashop.kc_model import (
    fill_kc_template,
    load_kc_template,
    save_kc_template,
    template_models,
    template_universe,
)
from kcluster.paths import run_dir


def main(args):
    run = run_dir(args.run_dir)
    kc_paths = [os.path.abspath(path) for path in args.kc_paths]
    kc_paths += run_dir_kc_paths(run) if run else []
    if not kc_paths:
        raise SystemExit("No KC models to add: pass --run_dir (or set KCLUSTER_RUN_DIR) or --kc_paths")
    kc_models = load_kc_models(dict.fromkeys(kc_paths))

    template = load_kc_template(args.kc_temp)
    if own := template_models(template):
        print(f"*** The template's KC models set the steps to label: {', '.join(own)} ***")
    filled = fill_kc_template(template, kc_models, restrict_suffix=args.restrict_suffix)
    print(f"*** Labelled {int(template_universe(filled).sum()):,} of {len(filled):,} steps ***")

    output = args.output or _default_output(args.kc_temp, run)
    save_kc_template(filled, output)
    print(f"*** Wrote {len(template_models(filled))} KC models to {output} ***")


def _default_output(kc_temp: str, run: str | None) -> str:
    """``<template>-import.txt``, in the run dir whose models it carries, else beside the template."""
    name = f"{os.path.splitext(os.path.basename(kc_temp))[0]}-import.txt"
    return os.path.join(run or os.path.dirname(os.path.abspath(kc_temp)), name)


def add_arguments(parser):
    parser.add_argument("--kc_temp", required=True, type=str,
                        help="A KC-model export from DataShop to fill. Its KC models set the steps to label: "
                             "every model added must label each step they all tag (every step, if it has none), "
                             "and none other. They are left out of the output, since importing them again "
                             "would overwrite them.")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="A result dir whose kc/*-kc.csv models are all added — or, for a work dir "
                             "of per-bank result dirs, each of their kc/*-kc.csv")
    parser.add_argument("--kc_paths", nargs="*", default=[], type=str,
                        help="KC CSVs to add (<ds>_<model>-kc.csv; the model becomes the column name)")
    parser.add_argument("--restrict_suffix", default=None, type=str,
                        help="For a question bank that reaches fewer steps than the template's KC models tag: "
                             "label only the steps the questions reach, and add each of the template's models "
                             "again as <model>-<suffix>, cut down to those steps, so DataShop compares all of "
                             "them over the same steps (e.g. MCQ gives LOs-MCQ beside LOs)")
    parser.add_argument("--output", default=None, type=str,
                        help="Output path (default: <template name>-import.txt in --run_dir, else beside "
                             "the template)")
