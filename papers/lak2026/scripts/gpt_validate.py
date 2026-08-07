"""GPT external-judge arm: OpenAI Batch API requests + response filtering.

``--mode prepare`` writes per-directory batch_reqs.jsonl files (and a
deduplicated all_batch_reqs.jsonl) for every validated question set;
``--mode validate`` reads the returned batch_output-*.jsonl files and keeps
the questions whose answer GPT agrees with.
"""

import argparse
import glob
import itertools
import json
import os

import pandas as pd

from kcluster.core.prompts import GPT_JUDGE_SYSTEM_PROMPT
from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, load_questions

PROMPT_SEP = "\n\n\n\n\n/br/"


def prepare_batch_requests(questions: list[Question], model: str = "gpt-4o-mini"):
    sys_prompt = GPT_JUDGE_SYSTEM_PROMPT
    json_schema = {
        "name": "mcq_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "final_answer": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["final_answer", "explanation"],
            "additionalProperties": False}
    }

    batch_requests = []
    for q in questions:
        req = {
            "custom_id": q["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "system", "content": sys_prompt},
                             {"role": "user", "content": q.prompt()}],
                "max_completion_tokens": 100,
                "response_format": {"type": "json_schema", "json_schema": json_schema}
            }
        }
        batch_requests.append(req)

    return batch_requests


def write_batch_requests(root_dir: str, model: str = "gpt-4o-mini"):
    """Write batch_reqs.jsonl next to every validated question set, then a
    deduplicated all_batch_reqs.jsonl at the root."""
    for fname in sorted(glob.iglob("**/new-mcq-valid.jsonl", recursive=True, root_dir=root_dir)):
        questions = load_questions(os.path.join(root_dir, fname))
        batch_reqs = prepare_batch_requests(questions, model=model)
        with open(os.path.join(root_dir, os.path.split(fname)[0], "batch_reqs.jsonl"), "w") as f:
            for req in batch_reqs:
                f.write(json.dumps(req) + "\n")

    # Merge all batch requests
    all_reqs, all_ids = list(), set()
    for fname in glob.iglob("**/batch_reqs.jsonl", recursive=True, root_dir=root_dir):
        with open(os.path.join(root_dir, fname), "r") as f:
            for line in f:
                req = json.loads(line)
                if req["custom_id"] not in all_ids:
                    all_ids.add(req["custom_id"])
                    all_reqs.append(req)

    with open(os.path.join(root_dir, "all_batch_reqs.jsonl"), "w") as f:
        for req in all_reqs:
            f.write(json.dumps(req) + "\n")
    print(f"Wrote {len(all_reqs)} deduplicated requests to all_batch_reqs.jsonl")


def gpt_validate(root_dir: str):
    """Validate MCQs using GPT responses"""

    # Load GPT responses
    qid_to_ans = dict()
    for fname in glob.iglob("batch_output-*.jsonl", root_dir=root_dir):
        print(f"Reading {fname}")
        with open(os.path.join(root_dir, fname), "r") as f:
            for line in f:
                try:
                    rsp = json.loads(line)
                    ans = json.loads(rsp["response"]["body"]["choices"][0]["message"]["content"])
                except ValueError:
                    content = rsp["response"]["body"]["choices"][0]["message"]["content"].rstrip('"}') + '"}'
                    ans = json.loads(content)
                finally:
                    q_id = rsp["custom_id"]
                    ans = ans["final_answer"].strip().lower()
                    assert ans in tuple("abcde"), f"Invalid answer: '{ans}'"
                    qid_to_ans[q_id] = ans

    for fname in sorted(glob.iglob("**/new-mcq-valid.jsonl", recursive=True, root_dir=root_dir)):
        curr_dir = os.path.join(root_dir, os.path.split(fname)[0])
        output_dir = os.path.join(curr_dir, "gpt-validated")
        os.makedirs(output_dir, exist_ok=True)

        # Read all questions
        all_questions = load_questions(os.path.join(root_dir, fname))

        # Read all prompts
        with open(os.path.join(curr_dir, "prompts-valid.txt"), "r") as f:
            all_prompts = [p.strip() for p in f.read().split("/br/")]

        mask = [qid_to_ans[q["id"]] == q.answer for q in all_questions]
        valid_questions = list(itertools.compress(all_questions, mask))
        valid_prompts = list(itertools.compress(all_prompts, mask))
        assert len(valid_questions) == len(valid_prompts), "Number of valid questions and prompts do not match"
        print(f"Found {len(valid_questions)} valid questions for {fname}")
        print(f"*** valid rate = {len(valid_questions) / len(all_questions):.2%} ***")

        # Save questions
        dump_questions(valid_questions, os.path.join(output_dir, "new-mcq-gpt-valid.jsonl"))
        res_df = pd.DataFrame.from_records(q.flat_dict for q in valid_questions)
        res_df.to_csv(os.path.join(output_dir, "new-mcq-gpt-valid.csv"), index=False)

        # Save prompts
        with open(os.path.join(output_dir, "prompts-gpt-valid.txt"), "w") as f:
            f.write(PROMPT_SEP.join(p for p in valid_prompts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prepare", "validate"),
                        help="prepare: write Batch API requests; validate: filter by returned answers")
    parser.add_argument("--root_dir", required=True, type=str,
                        help="Root directory of validated question sets (and batch_output-*.jsonl for validate)")
    parser.add_argument("--model", default="gpt-4o-mini", type=str, help="OpenAI model for prepared requests")
    args = parser.parse_args()

    if args.mode == "prepare":
        write_batch_requests(args.root_dir, model=args.model)
    else:
        gpt_validate(args.root_dir)
