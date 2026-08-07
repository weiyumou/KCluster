"""Concept extraction: naming the key concept a question tests (EDM 2025).

The ``Remark:`` prompt asks the model to complete "...tests whether the
student understands the concept of" (noun phrase, the default) or "...tests
whether the student can" (verb phrase, ``verbal=True``). Beam search with a
negative length penalty keeps completions short, and generation stops at the
first period or comma. Clustering later propagates these concept labels from
cluster exemplars to their members.
"""

import torch
from tqdm import tqdm

from kcluster.core.prompts import concept_prompt, congruity_marginal_context
from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel, batched


@torch.inference_mode()
def extract_concepts(llm: LargeLangModel, questions: list[Question],
                     batch_size: int, verbal: bool = False, **kwargs) -> list[str]:
    """Extracts the key concept for a list of Questions"""
    all_concepts = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting concepts"):
        prompts = [concept_prompt(q, verbal=verbal) for q in batch]
        concepts = llm.complete_prompts(prompts, stop_tokens=[".", ","], **kwargs)
        all_concepts.extend([c.strip().rstrip(".,") for c in concepts])

    return all_concepts


@torch.inference_mode()
def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int, **kwargs) -> torch.Tensor:
    all_embeddings = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting question embeddings"):
        contexts = [congruity_marginal_context(q) for q in batch]
        texts = [str(q) for q in batch]
        all_embeddings.append(llm.encode(texts, contexts, **kwargs))

    return torch.cat(all_embeddings, dim=0)


