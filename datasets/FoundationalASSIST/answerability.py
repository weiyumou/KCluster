"""Screen Foundational ASSIST problems for answerability with Gemini.

Some problems cannot be answered from their own text: they refer to a figure
that is not described, or to a quantity introduced in an earlier part of the
problem set that the export does not ship. This script asks Gemini to answer
every problem and records two independent signals per problem:

* **answer arm** — the model's answer, compared to the key by *exact* match.
  Select-1 problems go through the prefilled-turn log-probability trick (the
  first generated token is the choice letter), so they also yield a full
  distribution over the choices, with a "None of the above" option appended.
  Select-all and fill-in problems are answered as text.
* **probe arm** — a direct Yes/No question ("can this be answered from the
  information it contains?"), also read from choice log-probabilities.

Treat a *correct* answer as evidence of answerability and a *failure* as a
candidate for manual review, not as grounds for automatic removal: a failure
conflates missing context with a genuinely hard problem, a tolerance-graded key
(``9pi``, ``40000``), and a key error. Nothing here modifies the dataset — the
output is a flag table for review.

The prompts live in this script rather than ``kcluster.core.prompts`` because
they are dataset-cleaning tools, not published artifacts of either paper.

Requires google-genai (``pip install -e ".[gemini]"``) and either
``GOOGLE_API_KEY`` or ``--vertexai`` with your own project.
"""

import argparse
import asyncio
import json
import math
import os
import string
import time

import pandas as pd

from kcluster.core.question import Question
from kcluster.engine.gemini import GeminiEngine, save_json_responses, sorted_choices
from kcluster.io.jsonl import load_questions

FILL_IN = "Fill-in-the-blank(s)"
SELECT_ALL = "Multiple Choice (select all)"
SELECT_ONE = "Multiple Choice (select 1)"

# --- Prompts -------------------------------------------------------------
ANSWER_MCQ = "Answer the following question.\n\n{question}"
ANSWER_PREFILL = "The answer is **"

FIB_SYSTEM = (
    "You are answering a middle-school mathematics problem. Reply with the final answer only: no working, "
    "no explanation, and no units unless the question explicitly asks for them. If the problem has several "
    "blanks, give one answer per blank, in order, separated by commas."
)
SELECT_ALL_SYSTEM = (
    "You are answering a select-all-that-apply question. Reply with the letters of every correct option, in "
    'alphabetical order, separated by commas (for example "a, c"). Output nothing else.'
)
ANSWER_TEXT = "{question}"
SELECT_ONE_SYSTEM = "Only output the letter of the correct answer choice."
PROBE_SYSTEM = "Only output Yes or No."

# The probe answers in words, not option letters: the questions carry their own
# a)/b) labels, so lettered probe options would collide with the question's.
PROBE = (
    "Between the <<< and >>> markers below is one question from a middle-school mathematics assignment.\n\n"
    "<<<\n{question}\n>>>\n\n"
    "Can that question be answered using only the information between the markers — without seeing any figure, "
    "table, diagram, or earlier part of the problem that is not shown there? Reply Yes or No."
)
PROBE_PREFILL = "My answer is **"
PROBE_CHOICES = ("Yes", "No", "yes", "no", "YES", "NO")


def logprob_config(thinking_budget: int) -> dict:
    return {
        "seed": 42,
        "temperature": 1.0,
        "max_output_tokens": 1,
        "thinking_config": {"thinking_budget": thinking_budget},
        "response_logprobs": True,
        "logprobs": 19,
    }


def text_config(thinking_budget: int, system_instruction: str) -> dict:
    return {
        "seed": 42,
        "temperature": 0.0,
        "max_output_tokens": thinking_budget + 64,
        "thinking_config": {"thinking_budget": thinking_budget, "include_thoughts": False},
        "system_instruction": system_instruction,
    }


def nota_label(q: Question) -> str:
    """The letter for an appended "None of the above" option."""
    labels = sorted(item["label"] for item in q.choices)
    return chr(ord(labels[-1]) + 1)


def answer_job(q: Question, thinking_budget: int, use_logprobs: bool) -> tuple[list, dict, str]:
    """Build the (contents, config, id) job that asks Gemini to answer ``q``."""
    if q.q_type == SELECT_ONE:
        body = f"{q.body}\n{nota_label(q)}) None of the above"
        user = {"role": "user", "parts": [{"text": ANSWER_MCQ.format(question=body)}]}
        if use_logprobs:
            # Prefilled model turn: the next token is the choice letter, so its
            # log-probabilities give the full distribution over the options.
            prefill = {"role": "model", "parts": [{"text": ANSWER_PREFILL}]}
            return [user, prefill], logprob_config(thinking_budget), q["id"]
        return [user], text_config(thinking_budget, SELECT_ONE_SYSTEM), q["id"]

    system = SELECT_ALL_SYSTEM if q.q_type == SELECT_ALL else FIB_SYSTEM
    contents = [{"role": "user", "parts": [{"text": ANSWER_TEXT.format(question=q.body)}]}]
    return contents, text_config(thinking_budget, system), q["id"]


def probe_job(q: Question, thinking_budget: int, use_logprobs: bool) -> tuple[list, dict, str]:
    """Build the job that asks whether ``q`` is answerable from its own text."""
    user = {"role": "user", "parts": [{"text": PROBE.format(question=q.body)}]}
    if use_logprobs:
        prefill = {"role": "model", "parts": [{"text": PROBE_PREFILL}]}
        return [user, prefill], logprob_config(thinking_budget), q["id"]
    return [user], text_config(thinking_budget, PROBE_SYSTEM), q["id"]


# --- Response parsing ----------------------------------------------------
def token_probs(rsp: dict, labels) -> dict[str, float]:
    """Probabilities of ``labels`` among the first generated token's candidates."""
    candidates = rsp["candidates"][0]["logprobs_result"]["top_candidates"][0]["candidates"]
    return {token: math.exp(-neg_logprob) for neg_logprob, token in sorted(sorted_choices(labels, candidates))}


def response_text(rsp: dict) -> str:
    return rsp["candidates"][0]["content"]["parts"][0]["text"].strip()


def parse_responses(responses: list[str]) -> dict[str, dict]:
    """Index raw response lines by id."""
    return {rsp["id"]: rsp for rsp in map(json.loads, responses)}


def normalize_letters(text: str) -> str:
    """Canonicalize a letter answer set: "b,a" and "B, A" both become "a, b"."""
    letters = sorted({ch for ch in text.lower() if ch in string.ascii_lowercase})
    return ", ".join(letters)


def fill_in_match(answer: str, key: str) -> bool:
    """Exact match for a fill-in answer, per blank.

    Only the delimiters are interpreted — ``" || "`` separates alternative
    acceptable answers and ``", "`` separates the blanks of a multi-blank
    problem. The values themselves must match character for character; no
    numeric tolerance, unit stripping, or fraction/decimal equivalence.
    """
    def blanks(text: str) -> tuple[str, ...]:
        return tuple(part.strip() for part in text.split(","))

    given = blanks(answer)
    return any(given == blanks(alt) for alt in key.split("||"))


def pick_letter(raw: str, labels) -> str:
    """The choice letter in a text answer: the whole reply if it is one, else the first valid letter."""
    cleaned = raw.strip().lower()
    if cleaned in set(labels):
        return cleaned
    found = [ch for ch in cleaned if ch in set(labels)]
    if not found:
        raise ValueError(f"no choice letter in {raw!r}")
    return found[0]


def score_answer(q: Question, rsp: dict, use_logprobs: bool = True) -> dict:
    """Compare Gemini's answer to the key. Returns the per-question result row."""
    out = {"model_answer": None, "model_answer_raw": None, "p_answer": None, "p_key": None,
           "p_nota": None, "exact_match": None, "status": "ok"}
    if rsp is None:
        out["status"] = "missing"
        return out
    if "error" in rsp:
        out["status"] = "api_error"
        return out

    try:
        if q.q_type == SELECT_ONE:
            nota = nota_label(q)
            labels = [item["label"] for item in q.choices] + [nota]
            if use_logprobs:
                probs = token_probs(rsp, labels)
                top = max(probs, key=probs.get)
                out["p_answer"] = probs[top]
                out["p_key"] = probs.get(q.answer, 0.0)
                out["p_nota"] = probs.get(nota, 0.0)
            else:
                # No distribution available: the letter is all we get.
                out["model_answer_raw"] = response_text(rsp)
                top = pick_letter(out["model_answer_raw"], labels)
                out["p_nota"] = 1.0 if top == nota else 0.0
            out["model_answer"] = top
            out["exact_match"] = top == q.answer
        elif q.q_type == SELECT_ALL:
            raw = response_text(rsp)
            out["model_answer_raw"] = raw
            out["model_answer"] = normalize_letters(raw)
            out["exact_match"] = out["model_answer"] == normalize_letters(q.answer)
        else:  # fill-in
            raw = response_text(rsp)
            out["model_answer_raw"] = raw
            out["model_answer"] = raw
            out["exact_match"] = fill_in_match(raw, str(q.answer))
    except (KeyError, IndexError, TypeError, ValueError, AssertionError):
        out["status"] = "unparseable"
    return out


def score_probe(rsp: dict, use_logprobs: bool = True) -> dict:
    """Self-containment: a probability with logprobs, a hard 0.0/1.0 without."""
    out = {"p_self_contained": None, "probe_status": "ok"}
    if rsp is None:
        out["probe_status"] = "missing"
        return out
    if "error" in rsp:
        out["probe_status"] = "api_error"
        return out
    try:
        if use_logprobs:
            probs = token_probs(rsp, PROBE_CHOICES)
            yes = sum(p for token, p in probs.items() if token.lower() == "yes")
            no = sum(p for token, p in probs.items() if token.lower() == "no")
            out["p_self_contained"] = yes / (yes + no) if (yes + no) else None
        else:
            answer = response_text(rsp).strip().lower()
            if answer.startswith("yes"):
                out["p_self_contained"] = 1.0
            elif answer.startswith("no"):
                out["p_self_contained"] = 0.0
            else:
                out["probe_status"] = "unparseable"
    except (KeyError, IndexError, TypeError, ValueError, AssertionError):
        out["probe_status"] = "unparseable"
    return out


def flag_reasons(row: dict, nota_thd: float, probe_thd: float) -> str:
    """Why this problem needs a human look (empty string = it passed)."""
    reasons = []
    if row["status"] != "ok":
        reasons.append(f"answer_{row['status']}")
    elif not row["exact_match"]:
        reasons.append("wrong_answer")
    if row.get("p_nota") is not None and row["p_nota"] >= nota_thd:
        reasons.append("none_of_the_above")
    if row["probe_status"] != "ok":
        reasons.append(f"probe_{row['probe_status']}")
    elif row["p_self_contained"] is not None and row["p_self_contained"] < probe_thd:
        reasons.append("not_self_contained")
    return ";".join(reasons)


# --- Runner --------------------------------------------------------------
async def gather_chunked(engine: GeminiEngine, jobs: list, batch_size: int, delay: float | None, desc: str):
    """Run jobs in bounded batches so a large screen does not open thousands of calls at once."""
    responses = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        label = f"{desc} [{start + 1}-{start + len(chunk)}/{len(jobs)}]"
        responses.extend(await engine.gather_responses(chunk, delay=delay, desc=label))
    return responses


def load_previous(resume_from: str | None, filename: str) -> dict[str, dict]:
    """Reusable responses from an earlier run: everything that is not an error."""
    if not resume_from:
        return {}
    path = os.path.join(resume_from, filename)
    if not os.path.isfile(path):
        print(f"** No responses to resume from at {path} **")
        return {}
    with open(path) as f:
        parsed = [json.loads(line) for line in f if line.strip()]
    return {rsp["id"]: rsp for rsp in parsed if "error" not in rsp}


async def run_arm(engine, questions, build_job, args, previous, name):
    """Run one arm over the questions that do not already have a usable response."""
    todo = [q for q in questions if q["id"] not in previous]
    print(f"** {name}: {len(previous)} reused, {len(todo)} to request **")
    responses = []
    if todo:
        jobs = [build_job(q) for q in todo]
        responses = await gather_chunked(engine, jobs, args.batch_size, args.delay, name)
    by_id = dict(previous) | parse_responses(responses)
    return by_id


LOGPROB_HELP = """\
Log-probabilities are only served by Vertex AI on the Gemini 2.5 family. As of this writing the
Developer API returns "Logprobs is not enabled" for every model, and Gemini 3.x rejects the option
on both endpoints. For the probability signal, run:

    --vertexai --project <gcp-project> --model gemini-2.5-flash

Otherwise the screen still works in text mode: answers are compared by exact match as usual, but
there is no choice distribution and self-containment is a hard Yes/No instead of a probability."""


async def supports_logprobs(engine: GeminiEngine, thinking_budget: int) -> tuple[bool, str]:
    """One cheap call that settles whether this model and endpoint return logprobs."""
    contents = [
        {"role": "user", "parts": [{"text": ANSWER_MCQ.format(question="What is 2+2?\na) 3\nb) 4")}]},
        {"role": "model", "parts": [{"text": ANSWER_PREFILL}]},
    ]
    rsp = json.loads(await engine.get_response(contents, logprob_config(thinking_budget), "preflight"))
    if "error" in rsp:
        return False, rsp["error"].replace("\n", " ")[:160]
    try:
        token_probs(rsp, {"a", "b"})
        return True, "logprobs available"
    except (KeyError, IndexError, TypeError, AssertionError):
        return False, "the response carried no logprobs_result"


async def main_async(args):
    questions = load_questions(args.data_path)
    print(f"** Loaded {len(questions)} questions from {args.data_path} **")

    if args.limit_per_type:
        kept, seen = [], {}
        for q in questions:
            seen[q.q_type] = seen.get(q.q_type, 0) + 1
            if seen[q.q_type] <= args.limit_per_type:
                kept.append(q)
        questions = kept
        print(f"** Limited to {len(questions)} questions ({args.limit_per_type} per type) **")

    if args.dry_run:
        shown = {}
        for q in questions:
            if shown.get(q.q_type):
                continue
            shown[q.q_type] = True
            contents, config, _ = answer_job(q, args.thinking_budget, not args.no_logprobs)
            print(f"\n{'=' * 70}\n{q.q_type}  ({q['id']}, key={q.answer!r})\n{'=' * 70}")
            print("--- answer arm ---")
            for turn in contents:
                print(f"[{turn['role']}] {turn['parts'][0]['text']}")
            print(f"[config] {config}")
            print("--- probe arm ---")
            print(f"[user] {probe_job(q, args.thinking_budget, not args.no_logprobs)[0][0]['parts'][0]['text']}")
        return

    # A screening run is a derived working artifact, not part of the contract:
    # it lands in interim/, the sibling of the processed/ file it screens.
    output_dir = os.path.abspath(args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(args.data_path)),
        "interim", "answerability", time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(output_dir, exist_ok=True)
    print(f"** Writing results to {output_dir} **")

    engine = GeminiEngine(args.model, vertexai=args.vertexai, project=args.project, location=args.location)

    # Settle the logprob question with one call rather than 5,000 failures.
    use_logprobs = not args.no_logprobs
    if use_logprobs:
        available, detail = await supports_logprobs(engine, args.thinking_budget)
        if available:
            print(f"** Preflight: {detail} on {args.model} **")
        elif args.require_logprobs:
            raise SystemExit(f"\n** Preflight failed: {detail} **\n\n{LOGPROB_HELP}")
        else:
            use_logprobs = False
            print(f"\n** Preflight: no log-probabilities ({detail}) **\n{LOGPROB_HELP}\n")
    mode = "logprob" if use_logprobs else "text"

    answers = await run_arm(engine, questions, lambda q: answer_job(q, args.thinking_budget, use_logprobs),
                            args, load_previous(args.resume_from, "answer-responses.jsonl"), "answer")
    save_json_responses([json.dumps(r) for r in answers.values()],
                        os.path.join(output_dir, "answer-responses.jsonl"))

    probes = {}
    if not args.skip_probe:
        probes = await run_arm(engine, questions, lambda q: probe_job(q, args.thinking_budget, use_logprobs),
                               args, load_previous(args.resume_from, "probe-responses.jsonl"), "probe")
        save_json_responses([json.dumps(r) for r in probes.values()],
                            os.path.join(output_dir, "probe-responses.jsonl"))

    rows = []
    for q in questions:
        row = {"id": q["id"], "type": q.q_type, "answer_type": q["answer_type"], "mode": mode,
               "problem_set_id": q["problem_set_id"], "problem_part": q["problem_part"], "key": q.answer}
        row |= score_answer(q, answers.get(q["id"]), use_logprobs)
        row |= (score_probe(probes.get(q["id"]), use_logprobs) if not args.skip_probe
                else {"p_self_contained": None, "probe_status": "skipped"})
        row["flag_reason"] = flag_reasons(row, args.nota_thd, args.probe_thd)
        row["flagged"] = bool(row["flag_reason"])
        row["stem"] = q.stem
        rows.append(row)

    report = pd.DataFrame(rows)
    report_path = os.path.join(output_dir, "answerability.csv")
    report.to_csv(report_path, index=False)

    print(f"\n** Flagged {int(report['flagged'].sum())} of {len(report)} problems **")
    print(report[report["flagged"]].groupby("type")["id"].count().to_string())
    print("\n** Flag reasons **")
    print(report.loc[report["flagged"], "flag_reason"].str.split(";").explode().value_counts().to_string())
    print(f"\n** Saved the report to {report_path} **")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_path", default="data/processed/foundational-assist.jsonl", type=str,
                        help="Question JSONL written by processing.py")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Output directory (default: data/interim/answerability/<timestamp>)")
    parser.add_argument("--model", default="gemini-3.6-flash", type=str,
                        help="Gemini model; pass gemini-3.1-pro-preview for a more accurate (and pricier) screen. "
                             "Only Vertex AI's 2.5 family serves log-probabilities")
    parser.add_argument("--vertexai", action="store_true", help="Use Vertex AI instead of the Developer API")
    parser.add_argument("--project", default=None, type=str, help="GCP project id (with --vertexai)")
    parser.add_argument("--location", default="us-central1", type=str, help="GCP location (with --vertexai)")
    parser.add_argument("--no_logprobs", action="store_true",
                        help="Skip the preflight and answer in text mode: no choice distribution, and "
                             "self-containment becomes a hard Yes/No")
    parser.add_argument("--require_logprobs", action="store_true",
                        help="Fail instead of falling back to text mode when log-probabilities are unavailable")
    parser.add_argument("--thinking_budget", default=1024, type=int, help="Per-call thinking token budget")
    parser.add_argument("--batch_size", default=50, type=int, help="Concurrent calls per batch")
    parser.add_argument("--delay", default=None, type=float, help="Optional per-call delay ramp (seconds)")
    parser.add_argument("--limit_per_type", default=None, type=int,
                        help="Smoke test: screen only the first N questions of each problem type")
    parser.add_argument("--dry_run", action="store_true", help="Print one prompt per type and exit; no API calls")
    parser.add_argument("--resume_from", default=None, type=str,
                        help="An earlier output directory; its non-error responses are reused")
    parser.add_argument("--skip_probe", action="store_true", help="Run the answer arm only")
    parser.add_argument("--nota_thd", default=0.5, type=float,
                        help="Flag a select-1 problem when P(None of the above) reaches this")
    parser.add_argument("--probe_thd", default=0.5, type=float,
                        help="Flag a problem when P(self-contained) falls below this")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
