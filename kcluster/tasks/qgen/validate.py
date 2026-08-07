"""MCQ validation: syntactic filtering + answer confidence (LAK 2026).

``validate_mcq`` first drops incomplete questions (trivial stems, missing or
duplicate or degenerate choices, "all/none of the above" artifacts), then
keeps a question only if one choice's average probability — over every
permutation of the choices, with a "None of the above" distractor appended —
clears ``prob_thd``, and re-keys the answer to that choice.
``sort_questions`` orders each LO's surviving questions by perplexity under
their seeding prompt.
"""

import itertools
import random
import re
import string
from operator import itemgetter

import torch

from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel, batched


def shuffle_choices(questions: list[Question]) -> None:
    """Shuffle the choices of each question in place"""
    for q in questions:
        ans_idx = ord(q["answerKey"]) - ord("a")
        ans = q["question"]["choices"][ans_idx]
        q["answerKey"] = None  # delete the answer key
        random.shuffle(q["question"]["choices"])
        for idx, chc in enumerate(q["question"]["choices"]):
            chc["label"] = chr(ord("a") + idx)  # Re-label each choice
            if chc is ans:
                q["answerKey"] = chc["label"]  # Update the answer key
        assert q["answerKey"] is not None, "The answer key is lost"


@torch.inference_mode()
def validate_mcq(llm: LargeLangModel, questions: dict[str, list[Question]], batch_size: int = 16,
                 prob_thd: float = 0.9, num_choices: int = 4, shuffle: bool = True) -> dict[str, list[Question]]:
    choices = [Question.SPACE + chc for chc in string.ascii_lowercase[:num_choices + 1]]
    chc_ids = list(itertools.chain.from_iterable(llm.tokenizer(choices)["input_ids"]))
    undesired = {"all of the above", "none of the above", "yes", "no", "true", "false"}

    # Filter questions
    for lo in list(questions):
        valid = []
        # First check if each question is complete
        for q in questions[lo]:
            q_choices = {chc["text"].lower().strip() for chc in q["question"]["choices"]}
            if any([
                # if the question has a trivial stem
                len(q.stem) < 10,
                # if there are invalid or duplicate choices
                len(q_choices) < num_choices,
                # if any choice is trivial
                any(len(chc) < 5 for chc in q_choices),
                # if the question has a trivial explanation
                len(q["explanation"]) < 10,
                # if any undesired choice is present
                len(q_choices & undesired) > 0,
                # if any choice starts with "both" or "neither"
                any(chc.startswith("both") or chc.startswith("neither") for chc in q_choices),
            ]):
                continue
            # only complete questions are kept
            valid.append(q)

        # Retain only valid questions
        questions[lo], valid = valid, []

        # Then check if any choice has high enough probability
        for q in questions[lo]:
            chc_texts = [item["text"] for item in q["question"]["choices"]]
            if shuffle:
                random.shuffle(chc_texts)

            prompts, col_inds = [], []
            for inds in itertools.permutations(range(len(chc_texts))):
                new_choices = [{"label": chr(ord("a") + j), "text": chc_texts[idx]} for j, idx in enumerate(inds)]
                new_choices.append({"label": chr(ord("a") + num_choices), "text": "None of the above"})
                q["question"]["choices"] = new_choices
                prompts.append(q.prompt())
                q["question"]["choices"].pop()
                col_inds.append(torch.argsort(torch.tensor(inds + (num_choices,)), dim=-1))
            row_inds = torch.arange(len(prompts)).repeat((len(chc_ids), 1)).T.to(llm.device)  # (P, C)
            col_inds = torch.stack(col_inds, dim=0).to(llm.device)  # (P, C)

            log_probs = []
            for batch in batched(prompts, batch_size):
                log_probs.append(torch.log_softmax(llm.next_logits(list(batch))[:, chc_ids], dim=-1))
            log_probs = torch.cat(log_probs, dim=0)  # (P, C)
            log_probs = log_probs[row_inds, col_inds]  # (P, C)

            # "none of the above" should never be the top choice
            if torch.argmax(log_probs, dim=-1).eq(num_choices).any(dim=-1):
                continue
            avg_probs = torch.logsumexp(log_probs, dim=0).exp() / len(prompts)  # (C,)
            val, ind = torch.max(avg_probs, dim=-1)
            if val.item() >= prob_thd:
                ans = chc_texts[ind.item()]
                # Update the answer key
                for item in q["question"]["choices"]:
                    if item["text"] == ans:
                        q["answerKey"] = item["label"]
                        break
                valid.append(q)

        # Retain only valid questions
        if valid:
            questions[lo] = valid
        else:
            del questions[lo]

    return questions


@torch.inference_mode()
def sort_questions(llm: LargeLangModel,
                   questions: dict[str, list[Question]], prompts: list[str],
                   batch_size: int = 16) -> tuple[dict[str, list[Question]], list[str]]:
    """Sort questions by their perplexity"""

    # Group prompts by LO
    def chunk_unequal(lst, sizes):
        """Split a list into chunks of unequal sizes"""
        it = iter(lst)
        for size in sizes:
            yield list(itertools.islice(it, size))

    prompts = dict(zip(questions.keys(), chunk_unequal(prompts, (len(questions[lo]) for lo in questions))))

    # Compute PPL for each question
    for lo in questions:
        all_ppl = []
        for batch in batched(zip(questions[lo], prompts[lo], strict=True), batch_size):
            qs, ps = list(zip(*batch))
            # Extract the seeding prompts
            ps = [re.match(r"(?s:.)+?(?=1\.)", p).group(0).strip() + f"\n1.{Question.SPACE}" for p in ps]
            ppl = llm.log_prob(texts=[str(q) for q in qs], contexts=ps, return_ppl=True)
            all_ppl.extend(ppl.tolist())

        questions[lo], prompts[lo] = list(
            zip(*((q, p) for q, p, _ in
                  sorted(zip(questions[lo], prompts[lo], all_ppl, strict=True), key=itemgetter(-1))))
        )

    # Flatten prompts
    prompts = [p for lo in prompts for p in prompts[lo]]

    return questions, prompts
