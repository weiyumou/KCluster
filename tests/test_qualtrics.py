"""Tests for the Qualtrics survey writer and the .qsf post-processor."""

import json

from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions
from kcluster.output.qualtrics import force_response, write_txt


def _question(qid="q-0", false_lo=None):
    data = {
        "id": qid,
        "type": "Multiple Choice",
        "question": {"stem": "What is the capital of France?",
                     "choices": [{"label": "a", "text": "Paris"}, {"label": "b", "text": "London"}]},
        "answerKey": "a",
        "lo": "name European capitals",
    }
    if false_lo:
        data["false_lo"] = false_lo
    return Question(data)


def test_write_txt_renders_blocks_with_lo_probe(tmp_path):
    write_txt([_question(), _question(qid="q-1", false_lo="balance chemical equations")], str(tmp_path))
    survey = (tmp_path / "survey.txt").read_text()

    assert survey.startswith("[[AdvancedFormat]]")
    assert "[[Block:Block-1]]" in survey and "[[Block:Block-2]]" in survey
    assert "[[ID:q-0]]" in survey and "[[ID:q-0-LO]]" in survey
    assert "B1-Q1. What is the capital of France?" in survey
    assert "[[Choice]]\nParis\n[[Choice]]\nLondon\n[[Choice]]\nNone of the above" in survey
    # The LO probe shows the true LO by default, the false LO when planted
    assert "B1-Q2. Does the above question test whether a student can <strong>name European capitals</strong>?" in survey
    assert "B2-Q2. Does the above question test whether a student can <strong>balance chemical equations</strong>?" in survey


def test_force_response_randomizes_blocks_and_forces_answers(tmp_path):
    questions = [_question()]
    question_path = tmp_path / "survey_questions.jsonl"
    dump_questions(questions, str(question_path))

    qsf = {
        "SurveyElements": [
            {"Element": "FL", "Payload": {"Flow": [{"Type": "Block", "FlowID": "FL_2"}]}},
            {"Element": "SQ", "Payload": {"DataExportTag": "q-0",
                                          "Validation": {"Settings": {}}}},
            {"Element": "SQ", "Payload": {"DataExportTag": "q-0-LO",
                                          "Validation": {"Settings": {}}}},
            {"Element": "SQ", "Payload": {"DataExportTag": "unrelated",
                                          "Validation": {"Settings": {}}}},
        ]
    }
    qsf_path = tmp_path / "survey.qsf"
    qsf_path.write_text(json.dumps(qsf))

    force_response(str(qsf_path), str(question_path))

    out = json.loads((tmp_path / "FR-survey.qsf").read_text())
    flow_elem, q1, q2, other = out["SurveyElements"]
    [randomizer] = flow_elem["Payload"]["Flow"]
    assert randomizer["Type"] == "BlockRandomizer"
    assert randomizer["FlowID"] == "FL_3"
    assert randomizer["SubSet"] == 1 and randomizer["EvenPresentation"] is True
    assert randomizer["Flow"] == [{"Type": "Block", "FlowID": "FL_2"}]

    # Both the MCQ and its LO probe are forced; unrelated questions untouched
    assert q1["Payload"]["Validation"]["Settings"] == {"ForceResponse": "ON", "ForceResponseType": "ON"}
    assert q2["Payload"]["Validation"]["Settings"] == {"ForceResponse": "ON", "ForceResponseType": "ON"}
    assert other["Payload"]["Validation"]["Settings"] == {}
