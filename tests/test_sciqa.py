import json

import pandas as pd

from kcluster.io.jsonl import load_questions
from kcluster.io.loaders.sciqa import evaluate_kc, write_questions


def _problem(question, choices, answer, skill, image=None):
    return {
        "image": image,
        "question": question,
        "choices": choices,
        "answer": answer,
        "skill": skill,
        "subject": "natural science",
        "topic": "physics",
        "category": "materials",
        "grade": "grade3",
    }


def test_write_questions_emits_valid_json_questions(tmp_path):
    problems = {
        "1": _problem("Which is the most flexible?", ["paper", "clay tile"], 0, "identify properties"),
        "2": _problem("Which is the hardest?", ["glass", "rubber"], 0, "identify properties"),
        "3": _problem("What is shown?", ["a", "b"], 0, "identify properties", image="img.png"),  # image: dropped
        "4": _problem("Pick one.", ["only"], 0, "identify properties"),  # too few choices: dropped
        "5": _problem("Rare skill?", ["x", "y"], 1, "rare skill"),  # below min_skill_cnt: dropped
    }
    data_path = tmp_path / "problems.json"
    data_path.write_text(json.dumps(problems))

    write_questions(str(data_path), str(tmp_path), min_choice_cnt=2, min_skill_cnt=2)

    # The output is canonical, schema-valid JSONL (the legacy writer emitted
    # Python repr; this pins the fix)
    questions = load_questions(str(tmp_path / "sciqa-skill-2.jsonl"))
    assert [q["id"] for q in questions] == ["sciqa-1", "sciqa-2"]
    assert questions[0]["question"]["choices"] == [
        {"label": "a", "text": "paper"},
        {"label": "b", "text": "clay tile"},
    ]
    assert questions[0]["answerKey"] == "a"
    assert questions[0]["skill"] == "identify properties"


def test_evaluate_kc_scores_csvs_against_the_skill_column(tmp_path):
    pd.DataFrame(
        {
            "skill": ["s1", "s1", "s2", "s2"],
            "KC": ["k1", "k1", "k2", "k2"],
        }
    ).to_csv(tmp_path / "pmi-kc.csv", index=False)

    res = evaluate_kc(str(tmp_path), random_kc=True, num_runs=3)
    assert res.loc["pmi (2 KCs)"].eq(1.0).all()
    assert "pmi-rand (2 KCs)" in res.index
