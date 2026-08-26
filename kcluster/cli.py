"""Unified command-line interface for KCluster.

Exposes each pipeline step as a subcommand of a single ``kcluster`` entry point,
e.g. ``kcluster concept ...`` or ``kcluster build-kc ...``. Command modules are
imported lazily so that ``kcluster`` and ``kcluster --help`` stay cheap and a
single command with heavy (or broken) dependencies cannot break the whole CLI.
"""

import argparse
import importlib
import sys

from kcluster import __version__

# (subcommand, module path, one-line help). Each module must expose
# ``add_arguments(parser)`` and ``main(args)``.
COMMANDS = [
    ("concept", "kcluster.commands.concept",
     "Extract concept labels from a jsonl of questions"),
    ("pmi", "kcluster.commands.pmi",
     "Compute pairwise question congruity (PMI) between questions"),
    ("build-kc", "kcluster.commands.build_kc",
     "Build KC models from extracted concepts and PMI values"),
    ("embed", "kcluster.commands.embed",
     "Add embedding-based cosine KC models to an existing result dir"),
    ("tag", "kcluster.commands.tag",
     "Tag a minimal student-step file with KC models and opportunity counts"),
    ("fit", "kcluster.commands.fit",
     "Fit student models (AFM/PFA) to a tagged file, one row per KC model"),
    ("build-datashop-kc", "kcluster.commands.build_datashop_kc",
     "Insert KC models into a DataShop KC template / student-step file"),
    ("refine-datashop-kc", "kcluster.commands.refine_datashop_kc",
     "Refine a DataShop KC model by splitting poorly-fitting KCs"),
    ("vertex-launch", "kcluster.commands.vertex_launch",
     "Launch Vertex AI batch jobs for concept extraction + congruity scoring"),
    ("vertex-retrieve", "kcluster.commands.vertex_retrieve",
     "Retrieve PMI similarity matrices for completed Vertex batch jobs"),
    ("vertex-build-kc", "kcluster.commands.vertex_build_kc",
     "Build KC models from completed Vertex batch jobs"),
    ("classify", "kcluster.commands.classify",
     "Align questions with learning objectives via rectangular PMI"),
    ("qgen-generate", "kcluster.commands.qgen_generate",
     "Generate MCQs incrementally from course standards"),
    ("qgen-validate", "kcluster.commands.qgen_validate",
     "Filter generated MCQs by completeness and answer confidence"),
    ("serve", "kcluster.commands.serve",
     "Serve a local model over HTTP for interactive experiments"),
]

_MODULES = {name: module for name, module, _ in COMMANDS}


def _build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kcluster",
        description="KCluster: an LLM-based clustering approach to KC discovery.",
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"kcluster {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, _, help_text in COMMANDS:
        # Listed for `kcluster --help`; real arguments are added lazily below.
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    top_parser = _build_top_parser()

    if not argv or argv[0] in ("-h", "--help"):
        top_parser.print_help()
        return
    if argv[0] in ("-V", "--version"):
        top_parser.parse_args(argv)
        return

    command = argv[0]
    module_path = _MODULES.get(command)
    if module_path is None:
        top_parser.error(f"invalid command: {command!r} "
                         f"(choose from {', '.join(_MODULES)})")

    module = importlib.import_module(module_path)
    cmd_parser = argparse.ArgumentParser(prog=f"kcluster {command}")
    module.add_arguments(cmd_parser)
    args = cmd_parser.parse_args(argv[1:])
    module.main(args)


if __name__ == "__main__":
    main()
