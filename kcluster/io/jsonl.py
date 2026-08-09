"""Canonical JSONL I/O for question data.

One JSON object per line, following the schema documented in the KCluster
README. This module is the single entry point for reading and writing
question files — the inline ``json.loads`` loops and, especially, the
``eval()``-based readers found in the legacy repos must not be reintroduced.
"""

import json
from collections.abc import Callable, Iterable

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


def load_questions(path: str, validate: bool = True) -> list[Question]:
    """Read a JSONL question file (one JSON object per line)."""
    return _load(path, json.loads, "JSON", validate)


def dump_questions(questions: Iterable[Question], path: str) -> None:
    """Write questions as JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q.data))
            f.write("\n")


def _load(path: str, parse: Callable, fmt: str, validate: bool) -> list[Question]:
    questions = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = parse(line)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"{path}:{lineno}: not a valid {fmt} line: {e}") from e
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
