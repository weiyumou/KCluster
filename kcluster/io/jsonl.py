"""Canonical JSONL I/O for question data.

One JSON object per line, following the schema documented in the KCluster
README. This module is the single entry point for reading and writing
question files — the inline ``json.loads`` loops and, especially, the
``eval()``-based readers found in the legacy repos must not be reintroduced.
"""

import json
from collections.abc import Iterable

from kcluster.core.question import Question


def validate_question(q: Question) -> None:
    """Raise ValueError if a question lacks the fields the pipelines rely on."""
    for field in ("id", "type", "question", "answerKey"):
        if field not in q:
            raise ValueError(f"question is missing required field {field!r}")
    if not isinstance(q["question"], dict) or not q["question"].get("stem"):
        raise ValueError(f"question {q['id']!r} needs a non-empty 'question.stem'")
    # Any type in the "Multiple Choice ..." family (select-1, select-all) must
    # ship choices; whatever the type, the choices present must be well formed.
    if not q.choices and q.q_type.startswith("Multiple Choice"):
        raise ValueError(f"multiple-choice question {q['id']!r} has no choices")
    for choice in q.choices:
        if not isinstance(choice, dict) or "label" not in choice or "text" not in choice:
            raise ValueError(f"question {q['id']!r} has a malformed choice: {choice!r}")
    # A figure is a marker in the text plus a file in `figures`, in the same
    # order; the two must agree or an image-taking engine would show the wrong
    # figure, or none, at a marker.
    figures = q.get("figures")
    if figures is not None and not (isinstance(figures, list) and all(isinstance(f, str) and f for f in figures)):
        raise ValueError(f"question {q['id']!r} needs 'figures' to be a list of file paths, got {figures!r}")
    if (n_markers := q.figure_count) != len(q.figures):
        raise ValueError(f"question {q['id']!r} shows {n_markers} figure marker(s) but lists "
                         f"{len(q.figures)} file(s) in 'figures'")


def load_questions(path: str, validate: bool = True) -> list[Question]:
    """Read a JSONL question file (one JSON object per line)."""
    questions = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: not a valid JSON line: {e}") from e
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{lineno}: expected an object per line, got {type(data).__name__}")
            q = Question(data)
            if validate:
                try:
                    validate_question(q)
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e
            questions.append(q)
    return questions


def dump_questions(questions: Iterable[Question], path: str) -> None:
    """Write questions as JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.data))
            f.write("\n")
