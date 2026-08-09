"""Tests for the congruity format-leakage probe.

Two things have to hold or the probe is worthless: the renderers must strip
exactly the channel they name (and v1 must stay byte-identical to the published
grid), and the two AUCs must actually respond to a planted effect. The latter
is checked by building similarity matrices with a known format effect and a
known content effect and confirming each metric moves only for its own.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from kcluster.core import prompts
from kcluster.core.question import Question

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = REPO_ROOT / "datasets" / "FoundationalASSIST"


def _load(name: str):
    if str(DRIVER_DIR) not in sys.path:
        sys.path.insert(0, str(DRIVER_DIR))
    spec = importlib.util.spec_from_file_location(f"fa_{name}", DRIVER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load("format_probe")


def _mcq(qid="q1") -> Question:
    return Question({"id": qid, "type": "Multiple Choice (select 1)",
                     "question": {"stem": "What is 2+2?",
                                  "choices": [{"label": "a", "text": "3"}, {"label": "b", "text": "4"}]},
                     "answerKey": "b"})


def _fib(qid="q2") -> Question:
    return Question({"id": qid, "type": "Fill-in-the-blank(s)",
                     "question": {"stem": "28 is 7 times what? ____"}, "answerKey": "4"})


# --- renderers -------------------------------------------------------------
def test_v1_is_unchanged_by_the_render_switch():
    # The default path must be byte-identical to the published grid; the golden
    # tests pin the strings, this pins that routing through a renderer is a no-op.
    q1, q2 = _mcq(), _fib()
    assert prompts.congruity_pair_context(q1, q2) == prompts.congruity_pair_context(q1, q2, "v1")
    assert prompts.congruity_pair_context(q1, q2) == (
        f"{q1.header(1)}\n{q1}\n\n{q2.header(2)}\n")
    assert prompts.congruity_marginal_context(q2) == prompts.congruity_marginal_context(q2, "v1")
    assert prompts.congruity_scored_text(q2) == str(q2)


@pytest.mark.parametrize("render, has_type, has_answer", [
    ("v1", True, True),
    ("no_type", False, True),
    ("no_answer", False, False),
])
def test_each_renderer_strips_exactly_its_channel(render, has_type, has_answer):
    q1, q2 = _mcq(), _fib()
    ctx = prompts.congruity_pair_context(q1, q2, render)
    scored = prompts.congruity_scored_text(q2, render)
    # The type line of BOTH questions goes, or neither.
    assert ("Multiple Choice (select 1):" in ctx) is has_type
    assert ("Fill-in-the-blank(s):" in ctx) is has_type
    # The answer trailer goes from the context question and the scored text alike:
    # leaving it on either one keeps the cue in play.
    assert ("Answer: b" in ctx) is has_answer
    assert ("Answer: 4" in scored) is has_answer
    # The exercise headers and the question content survive every variant.
    assert ctx.startswith("Exercise 1:\n") and "\n\nExercise 2:\n" in ctx
    assert "What is 2+2?\na) 3\nb) 4" in ctx and "28 is 7 times what? ____" in scored


def test_unknown_renderer_is_rejected():
    with pytest.raises(ValueError, match="unknown congruity renderer"):
        prompts.congruity_marginal_context(_mcq(), "no-such-variant")


def test_pair_question_threads_the_renderer_through(probe):
    pytest.importorskip("torch")
    from kcluster.tasks.congruity import PairQuestion

    qs = [_mcq("a"), _fib("b")]
    assert PairQuestion(qs)[0] == PairQuestion(qs, render="v1")[0]
    ctx, text = PairQuestion(qs, render="no_answer")[2 + 0 * 2 + 1]
    assert "Answer:" not in ctx and "Answer:" not in text


# --- sampling --------------------------------------------------------------
def _cell(code, q_type, n):
    return [Question({"id": f"{code}-{q_type[:2]}-{i}", "type": q_type, "skill_code": [code],
                      "question": {"stem": f"stem {i}", "choices": [{"label": "a", "text": "x"}]}
                      if q_type.startswith("Multiple") else {"stem": f"stem {i}"},
                      "answerKey": "a"}) for i in range(n)]


def test_only_fully_crossed_codes_are_sampled(probe):
    quota = {probe.FILL_IN: 2, probe.SELECT_ONE: 2, probe.SELECT_ALL: 1}
    questions = []
    for code, counts in [("A.1", (3, 3, 2)), ("B.2", (3, 3, 2)), ("C.3", (9, 0, 0))]:
        for fmt, n in zip(probe.FORMATS, counts, strict=True):
            questions.extend(_cell(code, fmt, n))

    sample = probe.crossed_sample(questions, quota, seed=1)
    # C.3 is single-format: content and format are indistinguishable there, so
    # it must not enter the design however many questions it has.
    assert {probe.primary_code(q) for q in sample} == {"A.1", "B.2"}
    assert len(sample) == 2 * sum(quota.values())
    # Exactly the quota per cell, and the draw is reproducible.
    for code in ("A.1", "B.2"):
        for fmt in probe.FORMATS:
            assert sum(probe.primary_code(q) == code and q.q_type == fmt for q in sample) == quota[fmt]
    assert [q["id"] for q in sample] == [q["id"] for q in probe.crossed_sample(questions, quota, seed=1)]


def test_an_impossible_quota_fails_loudly(probe):
    questions = _cell("A.1", probe.FILL_IN, 5)
    with pytest.raises(SystemExit, match="No CCSS code carries every format"):
        probe.crossed_sample(questions, {probe.FILL_IN: 2, probe.SELECT_ONE: 2, probe.SELECT_ALL: 1})


# --- the metrics -----------------------------------------------------------
def test_auc_endpoints_and_ties(probe):
    assert probe.auc(np.array([3.0, 4.0, 1.0, 2.0]), np.array([True, True, False, False])) == 1.0
    assert probe.auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, False, False])) == 0.0
    assert probe.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([True, True, False, False])) == 0.5
    assert probe.auc(np.array([1.0, 2.0]), np.array([True, True])) is None  # one class only


def _synthetic(probe, monkeypatch, *, format_effect: float, content_effect: float):
    """A probe set plus a similarity matrix built from known effects."""
    codes, fmts, questions = [], [], []
    for code in ("A.1", "B.2", "C.3"):
        for fmt, k in zip(probe.FORMATS, (3, 3, 3), strict=True):
            for q in _cell(code, fmt, k):
                questions.append(q)
                codes.append(code)
                fmts.append(fmt)
    codes, fmts = np.array(codes), np.array(fmts)
    mat = (format_effect * (fmts[:, None] == fmts[None, :])
           + content_effect * (codes[:, None] == codes[None, :])).astype(float)

    monkeypatch.setattr(probe.PointwiseMutualInfo, "from_shards",
                        classmethod(lambda cls, *a, **k: type("P", (), {"pmi_mat": mat})()))
    return questions


def test_leak_detects_format_and_ignores_content(probe, monkeypatch):
    qs = _synthetic(probe, monkeypatch, format_effect=1.0, content_effect=0.0)
    res = probe.score_arm("unused", qs)
    assert res["leak"] == 1.0        # format perfectly separates content-unrelated pairs
    assert res["signal"] == 0.5      # and carries no content signal


def test_signal_detects_content_and_is_not_fooled_by_format(probe, monkeypatch):
    qs = _synthetic(probe, monkeypatch, format_effect=0.0, content_effect=1.0)
    res = probe.score_arm("unused", qs)
    assert res["signal"] == 1.0      # same-code pairs rank top among cross-format pairs
    assert res["leak"] == 0.5        # nothing left for format to explain


def test_a_pure_format_effect_cannot_masquerade_as_signal(probe, monkeypatch):
    # The point of restricting `signal` to cross-format pairs: a matrix driven
    # entirely by format must not look like content discrimination.
    qs = _synthetic(probe, monkeypatch, format_effect=5.0, content_effect=0.0)
    assert probe.score_arm("unused", qs)["signal"] == 0.5


def test_both_effects_are_separated_when_present_together(probe, monkeypatch):
    qs = _synthetic(probe, monkeypatch, format_effect=1.0, content_effect=1.0)
    res = probe.score_arm("unused", qs)
    assert res["signal"] == 1.0 and res["leak"] == 1.0
    # Unordered pairs only — counting (i, j) and (j, i) would double the sample.
    assert res["n_pairs"] == 27 * 26 // 2
