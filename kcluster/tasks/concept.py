"""Concept extraction: naming the key concept a question tests (EDM 2025).

The ``Remark:`` prompt asks the model to complete "...tests whether the
student understands the concept of" (noun phrase, the default) or "...tests
whether the student can" (verb phrase, ``verbal=True``). Beam search with a
negative length penalty keeps completions short, and generation stops at the
first period or comma. Clustering later propagates these concept labels from
cluster exemplars to their members.

``prompts`` is the prompt style (``kcluster.core.prompts``): the paper's plain
text unless a caller passes another, such as a chat style for an
instruction-tuned model. ``num_names`` asks for several names per question
instead of one: the model's best few under beam search, or draws under
sampling.
"""

from itertools import batched

import torch
from tqdm import tqdm

from kcluster.core.prompts import CONCEPT_STOP_TOKENS, STOCK, PromptStyle, clean_concept
from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel


@torch.inference_mode()
def extract_concepts(llm: LargeLangModel, questions: list[Question], batch_size: int, verbal: bool = False,
                     prompts: PromptStyle = STOCK, num_names: int | None = None, **kwargs) -> list:
    """The key concept of each question, or with ``num_names`` that many names of each.

    ``kwargs`` reach the model's ``generate``. With ``num_names`` the model
    returns that many completions per prompt (``num_return_sequences``; beam
    search needs at least as many beams, sampling ``do_sample=True``), and the
    result is a list of names per question, the completions that name nothing
    (no letter) left out; a question left with no name raises.
    """
    if num_names:
        kwargs["num_return_sequences"] = num_names
    all_concepts = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting concepts"):
        texts = [prompts.concept(q, verbal=verbal) for q in batch]
        concepts = [clean_concept(c) for c in llm.complete_prompts(texts, stop_tokens=list(CONCEPT_STOP_TOKENS), **kwargs)]
        if not num_names:
            all_concepts.extend(concepts)
            continue
        for q, group in zip(batch, batched(concepts, num_names)):
            names = [name for name in group if any(ch.isalpha() for ch in name)]
            if not names:
                raise ValueError(f"the model gave no name for question {q.get('id')}")
            all_concepts.append(names)

    return all_concepts


@torch.inference_mode()
def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int,
                            prompts: PromptStyle = STOCK, **kwargs) -> torch.Tensor:
    all_embeddings = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting question embeddings"):
        contexts = [prompts.marginal(q) for q in batch]
        texts = [str(q) for q in batch]
        all_embeddings.append(llm.encode(texts, contexts, **kwargs))

    return torch.cat(all_embeddings, dim=0)


