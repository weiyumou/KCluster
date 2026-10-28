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

The two tasks are thin over the general steps they share with any other
prompt: ``complete_names`` names what a batch of prompts asks about, and
``embed_texts`` encodes texts after their contexts. Both take prompts as the
engine does (``kcluster.engine``): strings, or conversations as messages.
"""

from collections.abc import Sequence
from itertools import batched

import torch
from tqdm import tqdm

from kcluster.core.prompts import CONCEPT_STOP_TOKENS, STOCK, PromptStyle, clean_concept
from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel, Prompt


@torch.inference_mode()
def complete_names(llm: LargeLangModel, prompts: Sequence[Prompt], batch_size: int,
                   stop_tokens: Sequence[str] = CONCEPT_STOP_TOKENS, num_names: int | None = None,
                   ids: Sequence | None = None, desc: str = "Naming", **kwargs) -> list:
    """The name each prompt asks the model to complete, or with ``num_names`` that many names per prompt.

    A name is the completion up to its first stop token or line break, cleaned (``clean_concept``). ``kwargs``
    reach the model's ``generate``. With ``num_names`` the model returns that many completions per prompt
    (``num_return_sequences``; beam search needs at least as many beams, sampling ``do_sample=True``), and the
    result is a list of names per prompt, the completions that name nothing (no letter) left out; a prompt left
    with no name raises, named by its entry in ``ids`` if given. A model that gives any other number of
    completions per prompt than asked for raises too, rather than pairing names with the wrong prompts.
    """
    if num_names:
        kwargs["num_return_sequences"] = num_names
    ids = ids if ids is not None else range(len(prompts))
    all_names = []
    for batch in tqdm(batched(zip(prompts, ids, strict=True), batch_size), desc=desc):
        texts = [prompt for prompt, _ in batch]
        completions = llm.complete_prompts(texts, stop_tokens=list(stop_tokens), **kwargs)
        if len(completions) != len(batch) * (num_names or 1):
            raise ValueError(f"the model gave {len(completions)} completions for {len(batch)} prompts, "
                             f"not {num_names or 1} per prompt")
        names = [clean_concept(c) for c in completions]
        if not num_names:
            all_names.extend(names)
            continue
        for (_, item), group in zip(batch, batched(names, num_names)):
            kept = [name for name in group if any(ch.isalpha() for ch in name)]
            if not kept:
                raise ValueError(f"the model gave no name for {item}")
            all_names.append(kept)

    return all_names


def extract_concepts(llm: LargeLangModel, questions: list[Question], batch_size: int, verbal: bool = False,
                     prompts: PromptStyle = STOCK, num_names: int | None = None, **kwargs) -> list:
    """The key concept of each question, or with ``num_names`` that many names of each (``complete_names``)."""
    return complete_names(llm, [prompts.concept(q, verbal=verbal) for q in questions], batch_size,
                          stop_tokens=prompts.concept_stop_tokens, num_names=num_names,
                          ids=[f"question {q.get('id')}" for q in questions], desc="Extracting concepts", **kwargs)


@torch.inference_mode()
def embed_texts(llm: LargeLangModel, texts: Sequence[str], contexts: Sequence[Prompt], batch_size: int,
                desc: str = "Embedding", **kwargs) -> torch.Tensor:
    """The engine's embedding of each text after its context, batched: (N, H)."""
    all_embeddings = []
    for batch in tqdm(batched(zip(texts, contexts, strict=True), batch_size), desc=desc):
        all_embeddings.append(llm.encode([t for t, _ in batch], [c for _, c in batch], **kwargs))

    return torch.cat(all_embeddings, dim=0)


def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int,
                            prompts: PromptStyle = STOCK, **kwargs) -> torch.Tensor:
    """Each question's text embedded after its marginal congruity context (``Question-emb`` in EDM 2025)."""
    return embed_texts(llm, [str(q) for q in questions], [prompts.marginal(q) for q in questions], batch_size,
                       desc="Extracting question embeddings", **kwargs)
