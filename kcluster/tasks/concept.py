"""Concept extraction: naming the key concept a question tests (EDM 2025).

The ``Remark:`` prompt asks the model to complete "...tests whether the
student understands the concept of" (noun phrase, the default) or "...tests
whether the student can" (verb phrase, ``verbal=True``). Beam search with a
negative length penalty keeps completions short, and generation stops at the
first period or comma. Clustering later propagates these concept labels from
cluster exemplars to their members.
"""

from itertools import batched

import torch
from tqdm import tqdm

from kcluster.core.prompts import (
    CONCEPT_STOP_TOKENS,
    clean_concept,
    concept_prompt,
    congruity_marginal_context,
)
from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel
from kcluster.io.figures import figure_paths


def _figures(batch, figure_dir: str | None) -> dict:
    """The ``figures`` argument for a batch, when any of it shows a figure (else nothing is passed)."""
    figures = [figure_paths(q, figure_dir) if figure_dir else [] for q in batch]
    return {"figures": figures} if any(figures) else {}


@torch.inference_mode()
def extract_concepts(llm: LargeLangModel, questions: list[Question],
                     batch_size: int, verbal: bool = False, figure_dir: str | None = None, **kwargs) -> list[str]:
    """Extracts the key concept for a list of Questions.

    ``figure_dir``: the root the questions' figure paths resolve against; the
    figures are then shown to the model. None shows none.
    """
    all_concepts = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting concepts"):
        prompts = [concept_prompt(q, verbal=verbal) for q in batch]
        concepts = llm.complete_prompts(prompts, stop_tokens=list(CONCEPT_STOP_TOKENS),
                                        **_figures(batch, figure_dir), **kwargs)
        all_concepts.extend([clean_concept(c) for c in concepts])

    return all_concepts


@torch.inference_mode()
def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int,
                            figure_dir: str | None = None, **kwargs) -> torch.Tensor:
    """Encodes each question under the marginal context (``figure_dir`` as in :func:`extract_concepts`)."""
    all_embeddings = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting question embeddings"):
        contexts = [congruity_marginal_context(q) for q in batch]
        texts = [str(q) for q in batch]
        all_embeddings.append(llm.encode(texts, contexts, **_figures(batch, figure_dir), **kwargs))

    return torch.cat(all_embeddings, dim=0)


