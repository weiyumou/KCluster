"""Where a question's figures are on disk.

A bank records each figure as a path relative to a *figure root*, by default
the folder the bank's JSONL sits in (the dataset contract, datasets/README.md).
Commands take ``--figure_dir`` to name another root, since a bank is often read
on a different machine than it was built on. These helpers resolve the paths
and check the files exist before a scoring job starts; opening the images is
the engine's business (``kcluster.engine.vision``).
"""

import os
from collections.abc import Iterable

from kcluster.core.question import Question


def figure_dir(data_path: str, explicit: str | None = None) -> str:
    """The figure root: ``--figure_dir`` when given, else the question file's folder."""
    return os.path.abspath(explicit or os.path.dirname(os.path.abspath(data_path)))


def figure_paths(question: Question, root: str) -> list[str]:
    """The question's figures as paths under ``root``, in marker order."""
    return [os.path.join(root, f) for f in question.figures]


def has_figures(questions: Iterable[Question]) -> bool:
    return any(q.figures for q in questions)


def add_arguments(parser) -> None:
    """The figure options of a scoring command (read back by :func:`figure_root`)."""
    parser.add_argument("--figure_dir", default=None, type=str,
                        help="Folder the bank's figure paths are relative to (default: the question file's folder)")
    parser.add_argument("--ignore_figures", action="store_true",
                        help="Show no figures: their markers reach the model as literal text (the text-only run)")


def figure_root(questions: list[Question], data_path: str, explicit_dir: str | None = None,
                ignore: bool = False) -> str | None:
    """The root to show a bank's figures from, or None when none are to be shown.

    None when the bank has no figures, or ``--ignore_figures`` asked for the
    text-only run; otherwise the files are checked first, so a missing one
    stops the command before the model loads.
    """
    n = sum(1 for q in questions if q.figures)
    if not n:
        return None
    if ignore:
        print(f"*** --ignore_figures: the {n} figure-bearing questions are scored as text ***")
        return None
    root = figure_dir(data_path, explicit_dir)
    try:
        check_figures(questions, root)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    print(f"*** Showing the figures of {n} questions, from {root} ***")
    return root


def check_figures(questions: Iterable[Question], root: str) -> int:
    """Raise unless every figure file exists under ``root``; returns how many questions show figures.

    Run before a scoring job, not inside it: a missing file would otherwise
    surface hours in, from a DataLoader worker.
    """
    missing, shown = [], 0
    for q in questions:
        if not q.figures:
            continue
        shown += 1
        missing.extend(p for p in figure_paths(q, root) if not os.path.isfile(p))
    if missing:
        listed = ", ".join(missing[:3]) + (f", ... ({len(missing)} in all)" if len(missing) > 3 else "")
        raise FileNotFoundError(f"figure file(s) not found under {root}: {listed} — pass --figure_dir "
                                "with the folder the bank's figure paths are relative to")
    return shown
