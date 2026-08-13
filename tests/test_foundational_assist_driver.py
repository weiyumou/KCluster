"""Tests for the Foundational ASSIST driver (datasets/FoundationalASSIST/).

Covers the parts that can fail silently: the HTML cleaning (vendored
``clean_utils`` plus the driver's own fixes), the problem-row -> Question
mapping, the truncated-decimal key repair, and the scoring of Gemini's
answerability responses.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from kcluster.core.question import Question

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_DIR = REPO_ROOT / "datasets" / "FoundationalASSIST"


def _load(name: str):
    # The drivers are run from their own directory and import each other by
    # bare name (processing.py -> unanswerable.py), so that has to be importable.
    if str(DRIVER_DIR) not in sys.path:
        sys.path.insert(0, str(DRIVER_DIR))
    spec = importlib.util.spec_from_file_location(f"fa_{name}", DRIVER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def screen():
    pytest.importorskip("google.genai")
    return _load("answerability")


@pytest.fixture(scope="module")
def processing():
    # The driver parses the export's HTML, so it needs a driver-only dependency
    # (the `datasets` group). A venv installed for *running* the pipeline —
    # `uv sync --extra local` on a cluster — does not have it, and these tests
    # must skip there rather than error, as the sibling driver tests do.
    pytest.importorskip("bs4")
    return _load("processing")


@pytest.fixture(scope="module")
def review_app():
    return _load("review_app")


def test_unanswerable_list_is_well_formed():
    mod = _load("unanswerable")
    ids = mod.UNANSWERABLE_PROBLEM_IDS
    assert len(ids) == len(set(ids)), "the drop list has duplicates"
    assert all(isinstance(i, int) and i > 0 for i in ids)
    assert ids == sorted(ids), "keep the list sorted so diffs stay readable"
    assert set(mod.REVIEWED_KEY_FIXES).isdisjoint(ids), "a corrected key belongs to a dropped problem"


def test_review_payload_never_contains_json_constants_the_browser_rejects(review_app):
    # json.dumps writes float('nan') as a bare NaN. Python's parser accepts it;
    # JSON.parse does not, and the page renders nothing. Regression test.
    payload = {"items": [{"p_key": float("nan"), "p_nota": float("inf"), "nested": [float("-inf")],
                          "keep": 0.5, "text": "fine", "none": None}]}
    encoded = review_app.dump_strict(payload)
    assert "NaN" not in encoded and "Infinity" not in encoded
    item = json.loads(encoded, parse_constant=lambda c: pytest.fail(f"bare {c} in payload"))["items"][0]
    assert item["p_key"] is None and item["p_nota"] is None and item["nested"] == [None]
    assert item["keep"] == 0.5 and item["text"] == "fine"


@pytest.mark.parametrize("key, expected", [
    ("-0.055555556", "-1/18"),      # 9-dp truncation of a repeating decimal
    ("-1.333333333", "-4/3"),
    ("-5.909090909", "-65/11"),
    ("0.000036", None),             # exact: 9/250 terminates, leave it alone
    ("0.000000564", None),          # exact despite 9 decimal places
    ("9.681", None),                # too few places to be machine truncation
    ("0.5", None),
    ("17160", None),                # not a decimal at all
])
def test_only_truncated_repeating_decimals_are_recovered(processing, key, expected):
    assert processing.recover_fraction(key) == expected


def _row(**overrides) -> pd.Series:
    row = {
        "problem_id": 151389, "Problem Set Id": "PSB6N4", "Problem Part": 1,
        "Problem Type": "Fill-in-the-blank(s)", "Answer Types": "Numeric",
        "Problem Body": "28 is 7 times what number? ____",
        "Fill-in Options": "4", "Fill-in Answers": "4",
        "Multiple Choice Options": "", "Multiple Choice Answers": "",
        "skill_code": "4.OA.A.1~~4.NF.B.4b", "skill_name": "Interpret Comparisons~~Multiply Fractions",
    }
    return pd.Series(row | overrides)


def test_fill_in_maps_answer_text_and_skills(processing):
    q = processing.build_question(_row())
    assert q["id"] == "fa-151389"
    assert q.q_type == "Fill-in-the-blank(s)"
    assert q.choices == []          # no choices: body is the stem alone
    assert q.answer == "4"
    assert q["skill"] == ["Interpret Comparisons", "Multiply Fractions"]
    assert q["skill_code"] == ["4.OA.A.1", "4.NF.B.4b"]
    assert q.body == "28 is 7 times what number? ____"


def test_select_one_maps_the_key_to_its_label(processing):
    q = processing.build_question(_row(**{
        "problem_id": 253, "Problem Type": "Multiple Choice (select 1)",
        "Answer Types": "Multiple Choice", "Problem Body": "Is there such a week?",
        "Fill-in Options": "", "Fill-in Answers": "",
        "Multiple Choice Options": "No || Yes", "Multiple Choice Answers": "Yes",
    }))
    assert [c["text"] for c in q.choices] == ["No", "Yes"]
    assert q.answer == "b"
    assert q.body == "Is there such a week?\na) No\nb) Yes"


def test_select_all_key_is_every_matching_label(processing):
    q = processing.build_question(_row(**{
        "problem_id": 211131, "Problem Type": "Multiple Choice (select all)",
        "Answer Types": "Check All That Apply", "Problem Body": "Select all.",
        "Fill-in Options": "", "Fill-in Answers": "",
        "Multiple Choice Options": "w || x || y || z", "Multiple Choice Answers": "z || x",
    }))
    assert q.answer == "b, d"


def test_answer_key_absent_from_options_is_rejected(processing):
    with pytest.raises(ValueError, match="not among the options"):
        processing.build_question(_row(**{
            "Problem Type": "Multiple Choice (select 1)", "Multiple Choice Options": "No || Yes",
            "Multiple Choice Answers": "Maybe",
        }))


# --- HTML cleaning ----------------------------------------------------------
# The vendored clean_utils absorbed a runtime patch of the export's own module;
# these pin the behaviors the patch existed for, verified equivalent to the
# patched original over every raw text cell (16,975; 0 disagreements).

@pytest.mark.parametrize("body, expected", [
    # nested structure survives: mfrac inside mfenced, not flattened to "14"
    ("<math><mfenced><mfrac><mn>1</mn><mn>4</mn></mfrac></mfenced></math>", "(1/4)"),
    # nth root: was rendering ∛27 as "273"
    ("<math><mroot><mn>27</mn><mn>3</mn></mroot></math>", "root3(27)"),
    # whitespace-only <mo> is a space between words…
    ("<math><mi>1</mi><mo>&#160;</mo><mi>to</mi><mo>&#160;</mo><mi>12</mi></math>", "1 to 12"),
    # …but nothing inside a spaced-out numeral
    ("<math><mn>0</mn><mo>.</mo><mo>&#160;</mo><mn>2</mn></math>", "0.2"),
    # upstream's own rendering is unchanged
    ("<math><mfrac><mn>4</mn><mn>3</mn></mfrac></math>", "(4/3)"),
    ("Area is <math><msup><mi>x</mi><mn>2</mn></msup></math>.", "Area is x^2 ."),
])
def test_vendored_mathml_parsing(processing, body, expected):
    assert processing.clean_utils.clean_problem_body(body) == expected


def test_clean_body_renders_blanks_and_html_scripts(processing):
    # the driver's own layer: <ast-r> answer blanks (numbered when multiple)
    # and HTML sup/sub, neither of which upstream handles
    clean_body = processing.make_clean_body()
    assert clean_body('1 - <ast-r marker="1"></ast-r> = 0.863') == "1 - ____ = 0.863"
    assert clean_body('<ast-r marker="1"/> and <ast-r marker="2"/>') == "____(1) and ____(2)"
    assert clean_body("10<sup>3</sup> cm<sup>-2</sup>") == "10^3 cm^(-2)"


# --- student-step reduction ------------------------------------------------
def _interactions() -> pd.DataFrame:
    # Student 7 attempts problem 151389 three times: ids 1 and 2 tie on the
    # earliest end_time, so id 1 (incorrect) must win the tie-break on log id
    # over id 2 (correct); id 3 is later. Timestamps carry the export's
    # stripped-zero precision on purpose.
    return pd.DataFrame({
        "id": [3, 1, 2, 4, 5],
        "problem_id": [151389, 151389, 151389, 253, 253],
        "user_id": [7, 7, 7, 7, 8],
        "hint_count": [0, 2, 0, 0, 0],
        "answer_text": ["4", "5", "4", "Yes", "No"],
        "saw_answer": [False, False, False, False, True],
        "discrete_score": [1.0, 0.0, 1.0, 1.0, 0.0],
        "end_time": ["2019-08-25 22:52:54.87+00", "2019-08-25 22:52:54+00",
                     "2019-08-25 22:52:54+00", "2019-08-26 02:07:21.1+00",
                     "2019-09-05 00:54:03.194+00"],
    })


def _problems() -> pd.DataFrame:
    return pd.DataFrame({"problem_id": [151389, 253],
                         "skill_code": ["4.OA.A.1~~4.NF.B.4b", "6.EE.B.5"]})


def test_first_attempt_is_earliest_end_time(processing):
    ss = processing.build_student_step(_interactions(), _problems())
    assert len(ss) == 3  # one row per (student, problem)
    row = ss[(ss["Anon Student Id"] == "7") & (ss["Step Name"] == "fa-151389")].squeeze()
    assert row["First Attempt"] == "incorrect"  # id 1, not the later corrects
    assert row["First Transaction Time"] == "2019-08-25 22:52:54+00"


def test_student_step_carries_ids_and_expert_kc(processing):
    ss = processing.build_student_step(_interactions(), _problems())
    assert ss["Problem Name"].equals(ss["Step Name"])
    fa253 = ss[ss["Step Name"] == "fa-253"]
    assert fa253["KC (CCSS)"].eq("6.EE.B.5").all()
    assert "Opportunity (CCSS)" not in ss.columns


def test_student_step_meets_the_contract(processing):
    from kcluster.io.student_step import check_coverage, validate_student_step
    problem_rows = [_row(), _row(**{
        "problem_id": 253, "Problem Type": "Multiple Choice (select 1)",
        "Answer Types": "Multiple Choice", "Problem Body": "Is there such a week?",
        "Fill-in Options": "", "Fill-in Answers": "",
        "Multiple Choice Options": "No || Yes", "Multiple Choice Answers": "Yes",
    })]
    questions = [processing.build_question(row) for row in problem_rows]
    ss = processing.build_student_step(_interactions(), _problems())
    validate_student_step(ss)
    assert check_coverage(questions, ss) == []


# --- answerability scoring -------------------------------------------------
def _logprob_response(tokens: dict) -> dict:
    candidates = [{"token": t, "log_probability": math.log(p)} for t, p in tokens.items()]
    return {"id": "q", "candidates": [{"logprobs_result": {"top_candidates": [{"candidates": candidates}]}}]}


def _text_response(text: str) -> dict:
    return {"id": "q", "candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture
def select_one() -> Question:
    return Question({"id": "q", "type": "Multiple Choice (select 1)",
                     "question": {"stem": "S?", "choices": [{"label": "a", "text": "No"},
                                                            {"label": "b", "text": "Yes"}]},
                     "answerKey": "b"})


def test_select_one_scores_the_full_choice_distribution(screen, select_one):
    assert screen.nota_label(select_one) == "c"  # appended None-of-the-above
    out = screen.score_answer(select_one, _logprob_response({"a": 0.05, "b": 0.9, "c": 0.05}))
    assert (out["model_answer"], out["exact_match"]) == ("b", True)
    assert out["p_key"] == pytest.approx(0.9) and out["p_nota"] == pytest.approx(0.05)


def test_none_of_the_above_winning_is_recorded(screen, select_one):
    out = screen.score_answer(select_one, _logprob_response({"a": 0.2, "b": 0.2, "c": 0.6}))
    assert out["exact_match"] is False and out["p_nota"] == pytest.approx(0.6)
    row = out | {"probe_status": "ok", "p_self_contained": 0.9}
    assert screen.flag_reasons(row, 0.5, 0.5) == "wrong_answer;none_of_the_above"


def test_select_all_compares_letter_sets_not_strings(screen):
    q = Question({"id": "q", "type": "Multiple Choice (select all)",
                  "question": {"stem": "S?", "choices": [{"label": ch, "text": ch} for ch in "abcd"]},
                  "answerKey": "b, d"})
    assert screen.score_answer(q, _text_response("D,B"))["exact_match"] is True
    assert screen.score_answer(q, _text_response("a, b"))["exact_match"] is False


@pytest.mark.parametrize("key, answer, expected", [
    ("1,000", "1,000", True),
    ("1,000", "1000", False),      # strict: no thousands-separator normalization
    ("316 || 314", "314", True),   # either alternative is accepted
    ("3, 4", "3,4", True),         # only the multi-blank delimiter is normalized
    ("3, 4", "4, 3", False),       # blanks are ordered
])
def test_fill_in_matching_is_strict_but_delimiter_aware(screen, key, answer, expected):
    q = Question({"id": "q", "type": "Fill-in-the-blank(s)", "question": {"stem": "S? ____"}, "answerKey": key})
    out = screen.score_answer(q, _text_response(answer))
    assert out["exact_match"] is expected
    assert out["model_answer_raw"] == answer


def test_probe_reads_yes_no_words_and_pools_casings(screen):
    assert screen.score_probe(_logprob_response({"Yes": 0.8, "No": 0.2}))["p_self_contained"] == pytest.approx(0.8)
    pooled = screen.score_probe(_logprob_response({"Yes": 0.5, "yes": 0.2, "No": 0.3}))
    assert pooled["p_self_contained"] == pytest.approx(0.7)


def test_failures_are_recorded_as_status_not_raised(screen, select_one):
    assert screen.score_answer(select_one, {"id": "q", "error": "429"})["status"] == "api_error"
    assert screen.score_answer(select_one, {"id": "q", "candidates": [{}]})["status"] == "unparseable"
    assert screen.score_answer(select_one, None)["status"] == "missing"
    assert screen.score_probe(_logprob_response({"maybe": 1.0}))["probe_status"] == "unparseable"


def test_a_clean_result_is_not_flagged(screen):
    row = {"status": "ok", "exact_match": True, "p_nota": 0.01,
           "probe_status": "ok", "p_self_contained": 0.99}
    assert screen.flag_reasons(row, 0.5, 0.5) == ""
