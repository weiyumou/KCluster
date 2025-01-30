import argparse
import json
import os
import re
import time

from tqdm import tqdm
from transformers import pipeline, set_seed
from transformers.utils import logging

from core.question import Question


def read_questions(data_path: str):
    # An iterator yielding questions read from a jsonl file
    with open(data_path, "r") as f:
        for line in f:
            q = Question(eval(line))
            yield [{"role": "system", "content": ""}, {"role": "user", "content": q.body}]


def main(args):
    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "cta", time.asctime())
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load an LLM
    pipe = pipeline("text-generation", model=args.llm_path, device_map="auto")

    # Run data through the pipeline
    responses = []
    for outputs in tqdm(pipe(read_questions(args.data_path),
                             max_new_tokens=1280, do_sample=True, top_p=0.95, temperature=0.6), desc="Deep thinking"):
        rsp = outputs[0]["generated_text"][-1]["content"]
        responses.append(re.search(r"<think>(.*?)</think>", rsp, re.DOTALL).group(1).strip())

    # Save results
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"{fname}-cta.txt"), "w") as f:
        f.write("\n".join(responses))

    # Save arguments
    with open(os.path.join(args.output_dir, f"args-{fname}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    set_seed(42)
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")

    cl_args = parser.parse_args()
    main(cl_args)
