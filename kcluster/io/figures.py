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
