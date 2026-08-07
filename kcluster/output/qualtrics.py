"""Qualtrics survey export (LAK 2026 expert study).

``write_txt`` renders questions in Qualtrics' Advanced TXT import format:
per question, one block with the MCQ itself (choices plus a "None of the
above" option) and a yes/no item asking whether the question tests its
learning objective (the ``false_lo`` field, when present, substitutes a
wrong LO — the study's manipulation). ``force_response`` post-processes the
.qsf exported after import: it wraps all blocks in an evenly-presenting
randomizer and forces a response on every survey question.
"""

import json
import os

from kcluster.core.question import Question
from kcluster.io.jsonl import load_questions


def write_txt(questions: list[Question], output_dir: str):
    SPACE = Question.SPACE
    with open(os.path.join(output_dir, "survey.txt"), "w") as f:
        f.write("[[AdvancedFormat]]\n\n")
        for idx, q in enumerate(questions, 1):
            lo = q.get("false_lo", q["lo"])
            lines = [
                f"[[Block:Block-{idx}]]\n\n",
                "[[Question:Text]]\n",
                "Answer the following two questions:\n\n",
                "[[Question:MC:SingleAnswer:Vertical]]\n",
                f"[[ID:{q['id']}]]\n",
                f"B{idx}-Q1.{SPACE}{q.stem}\n",
                "[[AdvancedChoices]]\n",
                "[[Choice]]\n",
                "\n[[Choice]]\n".join(item["text"] for item in q["question"]["choices"]),
                "\n[[Choice]]\nNone of the above\n\n",
                "[[Question:MC:SingleAnswer:Vertical]]\n",
                f"[[ID:{q['id']}-LO]]\n",
                f"B{idx}-Q2. Does the above question test whether a student can <strong>{lo}</strong>?\n",
                "[[AdvancedChoices]]\n",
                "[[Choice:1]]\nYes\n",
                "[[Choice:0]]\nNo\n\n\n"
            ]
            f.writelines(lines)


def force_response(qsf_path: str, question_path: str):
    # Load the .qsf file
    with open(qsf_path, "r") as f:
        qsf = json.load(f)

    # Load all survey questions and collect q_ids
    questions = load_questions(question_path)
    q_ids = set(q["id"] for q in questions) | set(f"{q['id']}-LO" for q in questions)

    for item in qsf["SurveyElements"]:
        # Randomize blocks
        if item.get("Element") == "FL":
            curr_flows = item["Payload"]["Flow"]
            flow_id = int(curr_flows[-1]["FlowID"].split("_")[1]) + 1
            item["Payload"]["Flow"] = [
                {
                    "Type": "BlockRandomizer", "FlowID": f"FL_{flow_id}",
                    "SubSet": len(curr_flows), "EvenPresentation": True,
                    "Flow": curr_flows,
                }
            ]

        # Force responses
        if (payload := item.get("Payload")) and (payload.get("DataExportTag") in q_ids):
            payload["Validation"]["Settings"].update(ForceResponse="ON", ForceResponseType="ON")

    save_path, save_file = os.path.split(qsf_path)
    with open(os.path.join(save_path, f"FR-{save_file}"), "w") as f:
        f.write(json.dumps(qsf))
