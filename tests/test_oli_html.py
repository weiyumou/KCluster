"""Tests for the OLI course-HTML loader, using synthetic OLI page markup."""

import pytest

bs4 = pytest.importorskip("bs4")

from kcluster.io.loaders.oli_html import parse_all_mcqs, parse_mcq  # noqa: E402


def _question_div(q_id="mcq-one", part_id="p1", stem_ps=("Which material bends?",),
                  choices=(("v1", "Paper (value: v1) [choice]"), ("v2", "Clay")),
                  responses=(("v1", "1"), ("v2", "0")), skillref=" skill-a ", img=None):
    body = "".join(f"<p>{p}</p>" for p in stem_ps)
    if img:
        body += f'<img src="{img}"/>'
    chc = "".join(f'<div value="{v}">{t}</div>' for v, t in choices)
    resp = "".join(
        f'<div class="oli-response" match="{m}"' + (f' score="{s}"' if s is not None else "") + "></div>"
        for m, s in responses
    )
    skill = f'<skillref idref="{skillref}"></skillref>' if skillref else ""
    return f"""
    <div class="oli-question" id="{q_id}">
      <div class="oli-body">{body}</div>
      <div class="oli-multiple-choice">{chc}</div>
      <div class="oli-part" id="{part_id}">{skill}{resp}</div>
    </div>
    """


def _write_html(path, *divs):
    path.write_text("<html><body>" + "".join(divs) + "</body></html>")
    return str(path)


def test_parse_mcq_extracts_a_complete_question(tmp_path):
    page = _write_html(
        tmp_path / "page.html",
        _question_div(stem_ps=("First line.", "Second line."), img="figs/pic.png"),
    )
    [q] = parse_mcq(page)

    assert q["id"] == "mcq-one"
    assert q["type"] == "Multiple Choice"
    assert q["question"]["stem"] == "First line.\nSecond line."
    # The "(value: ...)" and "[...]" annotations are stripped from choice text
    assert q["question"]["choices"] == [
        {"label": "a", "text": "Paper"},
        {"label": "b", "text": "Clay"},
    ]
    assert q["answerKey"] == "a"  # match="v1" with score 1 maps to the first choice
    assert q["images"] == ["pic.png"]
    assert q["oli-part-id"] == "p1"
    assert q["skillref"] == "skill-a"  # whitespace stripped
    assert q["step-name"] == "mcq-one_p1"


def test_parse_mcq_skips_malformed_questions(tmp_path):
    page = _write_html(
        tmp_path / "page.html",
        _question_div(q_id="no-stem", stem_ps=()),
        _question_div(q_id="empty-choice", choices=(("v1", "(value: v1)"), ("v2", "Clay"))),
        _question_div(q_id="multi-answer", responses=(("v1", "1"), ("v2", "1"))),
        _question_div(q_id="no-score", responses=(("v1", None), ("v2", "0"))),
        _question_div(q_id="unknown-answer", responses=(("v9", "1"), ("v2", "0"))),
        _question_div(q_id="ok"),
    )
    questions = parse_mcq(page)
    assert [q["id"] for q in questions] == ["ok"]


def test_parse_all_mcqs_dedupes_and_folds_identical_content(tmp_path):
    # page1 repeats in page2 verbatim (exact duplicate); page2 also carries the
    # same question under a different id/part (content duplicate to fold)
    same = _question_div(q_id="mcq-one", part_id="p1")
    _write_html(tmp_path / "page1.html", same, _question_div(q_id="mcq-two", part_id="p2",
                                                             stem_ps=("A different stem?",)))
    _write_html(tmp_path / "page2.html", same, _question_div(q_id="mcq-dup", part_id="p3",
                                                             skillref="skill-b"))

    questions = parse_all_mcqs(str(tmp_path))
    assert len(questions) == 2
    folded = next(q for q in questions if len(q["step-name"]) == 2)
    assert set(folded["step-name"]) == {"mcq-one_p1", "mcq-dup_p3"}
    assert set(folded["skillref"]) == {"skill-a", "skill-b"}
    assert all(q["id"].startswith("elearning-mcq-") for q in questions)
    assert "oli-part-id" not in folded


def test_parse_all_mcqs_id_prefix(tmp_path):
    _write_html(tmp_path / "page.html", _question_div())
    [q] = parse_all_mcqs(str(tmp_path), id_prefix="oli")
    assert q["id"] == "oli-0"
