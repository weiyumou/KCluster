"""Gemini judge with plain-text answers (the study's second judging mode).

Q1 answers the MCQ as a bare letter; Q2 judges the LO alignment as Yes/No.
The archived legacy variant of this script (gemini_new.py) also carried a
one-off retry list of question ids from a partially failed run; this
general form runs both questions for every input question.
"""

import argparse
import asyncio
import os
import time

import pandas as pd

from kcluster.core.question import Question
from kcluster.engine.gemini import GeminiEngine, parse_text_choices, save_json_responses
from kcluster.io.jsonl import load_questions

CONFIG = {
    "seed": 42,
    "temperature": 0.0,
    "max_output_tokens": 520,
    "thinking_config": {"thinking_budget": 512, "include_thoughts": False},
}

Q1_CONFIG = CONFIG | {
    "system_instruction": "Only output the letter of the correct answer choice.",
}

Q2_CONFIG = CONFIG | {
    "system_instruction": "Only output 'Yes' or 'No'.",
}

Q1_CHOICES = {"a", "b", "c", "d", "e"}
Q2_CHOICES = {"Yes", "No", "yes", "no"}


def prepare_q1_content(q: Question) -> list[dict]:
    ans_choices = sorted(item["label"] for item in q["question"]["choices"])
    next_choice = chr(ord(ans_choices[-1]) + 1)  # next letter after the last choice

    q_text = q.body + f"\n{next_choice}) None of the above"
    q1_user = {"role": "user", "parts": [{"text": f"Answer the following question:\n\n{q_text}"}]}
    return [q1_user]


def prepare_q2_content(q: Question) -> list[dict]:
    lo = q.get("false_lo", q["lo"])
    q2_user = {"role": "user",
               "parts": [{"text": f"Does the following question test whether a student can **{lo}**?\n\n{q.body}"}]
               }
    return [q2_user]


async def main_async(args):
    output_dir = getattr(args, "output_dir", None) or os.path.join(
        "results", "gemini", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=False)
    print(f"** Created output directory at {output_dir} **")

    engine = GeminiEngine(args.model, vertexai=args.vertexai, project=args.project, location=args.location)

    # Load questions
    questions = load_questions(args.data_path)
    print(f"** Loaded {len(questions)} questions from {args.data_path} **")

    # Q1: answer the MCQs
    q1_jobs = [(prepare_q1_content(q), Q1_CONFIG, q["id"]) for q in questions]
    q1_responses = await engine.gather_responses(q1_jobs, delay=args.delay, desc="Gathering Q1 responses")
    save_json_responses(q1_responses, os.path.join(output_dir, "q1-raw-responses.jsonl"))

    q1_results = parse_text_choices(q1_responses, Q1_CHOICES)
    pd.DataFrame.from_records(q1_results).to_csv(os.path.join(output_dir, "q1-answers.csv"), index=False)

    # Q2: judge the LO alignment
    q2_jobs = [(prepare_q2_content(q), Q2_CONFIG, q["id"]) for q in questions]
    q2_responses = await engine.gather_responses(q2_jobs, delay=args.delay, desc="Gathering Q2 responses")
    save_json_responses(q2_responses, os.path.join(output_dir, "q2-raw-responses.jsonl"))

    q2_results = parse_text_choices(q2_responses, Q2_CHOICES)
    pd.DataFrame.from_records(q2_results).to_csv(os.path.join(output_dir, "q2-answers.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to the input questions file (jsonl format)")
    parser.add_argument("--model", default="gemini-2.5-pro", type=str, help="Gemini model name")
    parser.add_argument("--vertexai", action="store_true", help="Use Vertex AI instead of the Developer API")
    parser.add_argument("--project", default=None, type=str, help="GCP project id (with --vertexai)")
    parser.add_argument("--location", default="us-central1", type=str, help="GCP location (with --vertexai)")
    parser.add_argument("--delay", default=0.5, type=float, help="Per-call delay ramp to respect rate limits")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")

    asyncio.run(main_async(parser.parse_args()))
