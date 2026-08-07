"""Gemini judge with prefilled-turn choice log-probabilities.

For each question: Q1 answers the MCQ (with a "None of the above" option
appended), Q2 judges whether the question tests its LO (the false_lo field,
when present, substitutes the study's manipulated LO). Both answers are read
from the log-probabilities of the first generated token after a prefilled
model turn ("The answer to Qn is **").
"""

import argparse
import asyncio
import os
import time

import pandas as pd

from kcluster.core.question import Question
from kcluster.engine.gemini import GeminiEngine, parse_logprob_choices, save_json_responses
from kcluster.io.jsonl import load_questions

Q1_CONFIG = {
    "seed": 42,
    "temperature": 1.0,
    "max_output_tokens": 1,
    "thinking_config": {"thinking_budget": 1024},
    "response_logprobs": True,
    "logprobs": 19
}
Q2_CONFIG = Q1_CONFIG

Q1_CHOICES = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}
Q2_CHOICES = {"a": "Yes", "b": "No"}


def prepare_q1_content(q: Question) -> list[dict]:
    ans_choices = sorted(item["label"] for item in q["question"]["choices"])
    next_choice = chr(ord(ans_choices[-1]) + 1)  # next letter after the last choice

    q_text = q.body + f"\n{next_choice}) None of the above"
    user = {"role": "user", "parts": [{"text": f"Answer the following two questions:\n\nQ1. {q_text}"}]}
    model = {"role": "model", "parts": [{"text": "The answer to Q1 is **"}]}
    return [user, model]


def prepare_q2_content(q: Question, q1_content: list[dict], q1_answer: str) -> list[dict]:
    lo = q.get("false_lo", q["lo"])
    q_text = (
        f"Does the above question help teachers test whether a student can {lo}?\n"
        "a) Yes\n"
        "b) No"
    )
    q2_user = {"role": "user", "parts": [{"text": f"Q2. {q_text}"}]}
    q2_model = {"role": "model", "parts": [{"text": "The answer to Q2 is **"}]}
    q1_user, q1_model = q1_content
    q1_model["parts"][0]["text"] += q["answerKey"] + "**."
    return [q1_user, q1_model, q2_user, q2_model]


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
    q1_responses = await engine.gather_responses(q1_jobs, desc="Gathering Q1 responses")
    save_json_responses(q1_responses, os.path.join(output_dir, "q1-raw-responses.jsonl"))

    q1_results = parse_logprob_choices(q1_responses, Q1_CHOICES)
    pd.DataFrame.from_records(q1_results).to_csv(os.path.join(output_dir, "q1-answers.csv"), index=False)

    # Q2: judge the LO alignment, conditioned on the answered Q1 turn
    q2_jobs = [(prepare_q2_content(q, prepare_q1_content(q), r["answer"]), Q2_CONFIG, q["id"])
               for q, r in zip(questions, q1_results)]
    q2_responses = await engine.gather_responses(q2_jobs, desc="Gathering Q2 responses")
    save_json_responses(q2_responses, os.path.join(output_dir, "q2-raw-responses.jsonl"))

    q2_results = parse_logprob_choices(q2_responses, Q2_CHOICES)
    pd.DataFrame.from_records(q2_results).to_csv(os.path.join(output_dir, "q2-answers.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to the input questions file (jsonl format)")
    parser.add_argument("--model", default="gemini-2.5-pro", type=str, help="Gemini model name")
    parser.add_argument("--vertexai", action="store_true", help="Use Vertex AI instead of the Developer API")
    parser.add_argument("--project", default=None, type=str, help="GCP project id (with --vertexai)")
    parser.add_argument("--location", default="us-central1", type=str, help="GCP location (with --vertexai)")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")

    asyncio.run(main_async(parser.parse_args()))
