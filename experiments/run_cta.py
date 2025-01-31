import argparse
import json
import os
import re
import time

from tqdm import tqdm
from transformers import pipeline, set_seed
from transformers.utils import logging

from core.model import batched
from core.question import Question


def read_questions(data_path: str):
    # An iterator yielding questions read from a jsonl file
    with open(data_path, "r") as f:
        for line in f:
            yield Question(eval(line))


def main(args):
    # Create a folder to store results
    if not hasattr(args, "output_dir"):
        args.output_dir = os.path.join("results", "cta", time.asctime())
    args.output_dir = args.output_dir.replace(' ', '_').replace(':', '-')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load an LLM
    pipe = pipeline("text-generation", model=args.llm_path, device_map="auto")
    end_think_id = pipe.tokenizer.convert_tokens_to_ids("</think>")

    # Run data through the pipeline
    fname = os.path.splitext(os.path.basename(args.data_path))[0]
    with open(os.path.join(args.output_dir, f"{fname}-cta.jsonl"), "w") as f_out:
        for batch in tqdm(batched(read_questions(args.data_path), args.batch_size), desc="Deep thinking"):
            prompts = [[{"role": "system", "content": ""}, {"role": "user", "content": q.body}] for q in batch]
            for idx, output in enumerate(pipe(prompts, eos_token_id=end_think_id,
                                              forced_bos_token_id=end_think_id, forced_eos_token_id=end_think_id,
                                              max_new_tokens=12800, top_p=0.95, temperature=0.6)):
                rsp = output[0]["generated_text"][-1]["content"]
                d = {"q_id": batch[idx]["id"], "thinking": ""}
                if m := re.search(r"<think>(.*?)</think>", rsp, re.DOTALL):
                    d["thinking"] = m.group(1).strip()
                f_out.write(json.dumps(d) + "\n")

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
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for processing questions")

    cl_args = parser.parse_args()
    main(cl_args)
