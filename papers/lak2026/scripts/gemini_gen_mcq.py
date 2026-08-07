"""Gemini MCQ generation baseline: structured-output batches per course."""

import argparse
import asyncio
import math
import os
import time

from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from kcluster.engine.gemini import GeminiEngine, save_json_responses


class MCQ(BaseModel):
    stem: str
    choices: list[str]
    answer: str


CONFIG = GenerateContentConfig(
    temperature=2.0,
    top_p=0.95,
    thinking_config={"thinking_budget": 768, "include_thoughts": False},
    max_output_tokens=1536,
    response_mime_type="application/json",
    response_schema=list[MCQ],
)


async def main_async(args):
    output_dir = getattr(args, "output_dir", None) or os.path.join(
        "results", "gemini-mcq", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=False)
    print(f"** Created output directory at {output_dir} **")

    engine = GeminiEngine(args.model, vertexai=args.vertexai, project=args.project, location=args.location)

    jobs = []
    for batch_idx in range(math.ceil(args.total / args.batch_size)):
        num_questions = min(args.batch_size, args.total - batch_idx * args.batch_size)
        prompt = (
            f"Generate {args.batch_size} multiple-choice questions (MCQ) suitable for a course titled, '{args.course}'."
        )
        if num_questions == 1:
            prompt = (
                f"Generate a multiple-choice question (MCQ) suitable for a course titled, '{args.course}'."
            )
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        jobs.append((contents, CONFIG, f"batch-{batch_idx}"))

    responses = await engine.gather_responses(jobs, delay=args.delay, desc="Gathering Responses")
    save_json_responses(responses, os.path.join(output_dir, "raw-responses.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", required=True, type=str, help="Course title for generating MCQs")
    parser.add_argument("--total", required=True, type=int, help="Total number of MCQs to generate")
    parser.add_argument("--batch_size", default=5, type=int, help="Number of MCQs to generate in a batch")
    parser.add_argument("--delay", default=None, type=float,
                        help="Delay in seconds between API calls to avoid rate limits")
    parser.add_argument("--model", default="gemini-2.5-flash", type=str, help="Gemini model name")
    parser.add_argument("--vertexai", action="store_true", help="Use Vertex AI instead of the Developer API")
    parser.add_argument("--project", default=None, type=str, help="GCP project id (with --vertexai)")
    parser.add_argument("--location", default="us-central1", type=str, help="GCP location (with --vertexai)")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="Path to the output directory")

    asyncio.run(main_async(parser.parse_args()))
