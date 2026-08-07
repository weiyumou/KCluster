"""Incremental MCQ generation from course standards (LAK 2026).

One MCQ is grown in steps, each conditioning on everything generated so
far: stem (optionally under classifier-free guidance against the bare
header), then each choice in label order, then the answer (next-token
choice over the label tokens), then the explanation. Standards come in two
grammatical shapes mirroring the classify task: ``actions`` (verb phrases)
and ``facts`` (statements).
"""

import hashlib
import itertools
import math
import re
import warnings
from collections import defaultdict

import torch
from tqdm import tqdm

from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel, batched


@torch.inference_mode()
def generate_mcq(llm: LargeLangModel, seed_prompts: list[str],
                 configs: dict[str, dict] = None, num_choices: int = 4) -> tuple[list[Question], list[str]]:
    """Generate MCQs from seed prompts subject to configs"""
    SPACE = Question.SPACE
    configs = defaultdict(dict, configs or {})
    begin_suppress_tokens = llm.tokenizer.encode("\n")

    # Step 1: create the stems
    new_stems = llm.complete_prompts(seed_prompts, stop_tokens=["\n"],
                                     begin_suppress_tokens=begin_suppress_tokens, **configs["stem"])
    # Repeat-interleave the prompts to match the length of new stems
    factor = len(new_stems) // len(seed_prompts)
    prompts = list(itertools.chain.from_iterable([p] * factor for p in seed_prompts))
    for i in range(len(new_stems)):
        new_stems[i] = re.match(r".+(?=\n)", new_stems[i] + "\n").group(0)
        prompts[i] += new_stems[i].rstrip().rstrip(";")
        new_stems[i] = new_stems[i].strip().rstrip(";")

    # Step 2: create the choices
    new_choices = [list() for _ in range(len(new_stems))]
    for idx in range(num_choices):
        opt = chr(ord("a") + idx)
        prompts = [f"{p}\n{opt})" for p in prompts]

        # Generate choices
        gen_choices = llm.complete_prompts(prompts, stop_tokens=["\n"],
                                           begin_suppress_tokens=begin_suppress_tokens, **configs["choice"])
        for i, gen_chc_str in enumerate(gen_choices):
            if (gen_chc := re.match(r".+?(?=[a-z]\))", gen_chc_str)) is None:
                gen_chc = re.match(r".+(?=\n)", gen_chc_str + "\n")
            gen_chc = gen_chc.group(0).strip().rstrip(".;")
            if gen_chc:
                gen_chc = gen_chc[0].upper() + gen_chc[1:]  # capitalize the initial

            new_choices[i].append((opt, gen_chc))
            prompts[i] += f"{SPACE}{gen_chc}"

    # Step 3: generate the answer
    prompts = [f"{p}\n\nSolution:\nThe correct answer is" for p in prompts]
    ans_tokens = [f"{SPACE}{chr(ord('a') + i)}" for i in range(num_choices)]
    answers = []
    for i, ans in enumerate(itertools.chain.from_iterable(llm.next_tokens(prompts, ans_tokens))):
        chc = new_choices[i][ans_tokens.index(ans)][-1]
        prompts[i] += f"{ans}){SPACE}{chc}."
        answers.append(ans.strip())

    # Step 4: generate explanation
    prompts = [f"{p}\n\nExplanation:\n" for p in prompts]
    explanations = llm.complete_prompts(prompts, stop_strings=["\n\n"], tokenizer=llm.tokenizer,
                                        begin_suppress_tokens=begin_suppress_tokens, **configs["explanation"])
    explanations = [re.match(r".+?(?=\n\n)", exp + "\n\n", re.DOTALL).group(0).strip() for exp in explanations]
    prompts = [f"{p}{e}" for p, e in zip(prompts, explanations, strict=True)]

    # Step 5: assemble parts to create questions
    new_questions = []
    for i in range(len(prompts)):
        choices = [{"label": lbl, "text": txt} for lbl, txt in new_choices[i]]
        q_dict = {
            "id": f"mcq-{i}",
            "type": "Multiple Choice",
            "question": {"stem": new_stems[i], "choices": choices},
            "answerKey": answers[i],
            "explanation": explanations[i]
        }
        new_q = Question(q_dict)
        new_q["id"] = hashlib.md5(str(new_q).encode("utf-8")).hexdigest()
        new_questions.append(new_q)

    return new_questions, prompts


def create_seed_prompts(standards: list[str], std_type: str, header: str) -> list[str]:
    match std_type:
        case "actions":
            return [
                f"The exercises below are designed to test whether a student can {std}.\n\n{header}"
                for std in standards]
        case "facts":
            return [
                f'The exercises below are designed to test whether a student understands the following facts:\n"{std}."\n\n{header}'
                for std in standards]
        case _:
            raise ValueError(f"Invalid std_type: '{std_type}'")


def read_standards(path: str, std_type: str) -> list[str]:
    """Read one standard per line, normalized like the classify task's LOs."""
    with open(path, "r") as f:
        match std_type:
            case "actions":
                return [line[0].lower() + line[1:].rstrip().rstrip(".") for line in f if line.strip()]
            case "facts":
                return [line.rstrip().rstrip(".") for line in f if line.strip()]
            case _:
                raise ValueError(f"Invalid std_type: '{std_type}'")


def generate_mcq_from_std(llm: LargeLangModel,
                          standards: list[str], std_type: str, stds_per_batch: int, qs_per_std: int,
                          configs: dict[str, dict] = None, num_choices: int = 4) -> tuple[list[Question], list[str]]:
    if not configs:
        warnings.warn("Given no custom config, default generation config is used")

    header = f"Multiple Choice (best out of {num_choices} options):\n1."

    all_questions, all_prompts = [], []
    for std_batch in tqdm(batched(standards, stds_per_batch), desc="Standards", leave=False):
        # Prepare the seeding prompts
        seed_prompts = create_seed_prompts(std_batch, std_type, header)

        bs = 8
        for j in range(math.ceil(qs_per_std / bs)):
            num_questions = min(bs, qs_per_std - j * bs)
            configs.setdefault("stem", {}).update(num_return_sequences=num_questions)

            # Enable classifier-free guidance (for stem only)
            if configs["stem"].get("guidance_scale", 1.0) > 1.0:
                factor = num_questions * len(seed_prompts)
                hdr = llm.tokenizer([header] * factor, return_tensors="pt").to(llm.device)
                configs["stem"]["negative_prompt_ids"] = hdr.input_ids
                configs["stem"]["negative_prompt_attention_mask"] = hdr.attention_mask

            questions, prompts = generate_mcq(llm, seed_prompts, configs, num_choices)
            for idx, q in enumerate(questions):
                q["lo"] = std_batch[idx // num_questions]
            all_questions.extend(questions)
            all_prompts.extend(prompts)

    return all_questions, all_prompts
