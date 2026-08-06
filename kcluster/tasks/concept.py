"""Concept extraction: naming the key concept a question tests (EDM 2025).

The ``Remark:`` prompt asks the model to complete "...tests whether the
student understands the concept of" (noun phrase, the default) or "...tests
whether the student can" (verb phrase, ``verbal=True``). Beam search with a
negative length penalty keeps completions short, and generation stops at the
first period or comma. Clustering later propagates these concept labels from
cluster exemplars to their members.
"""

import pandas as pd
import torch
from tqdm import tqdm

from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel, batched


@torch.inference_mode()
def extract_concepts(llm: LargeLangModel, questions: list[Question],
                     batch_size: int, verbal: bool = False, **kwargs) -> list[str]:
    """Extracts the key concept for a list of Questions"""
    SPACE = Question.SPACE

    # determine whether the generated concept should begin with a verb
    if verbal:
        trailer = "whether the student can"  # +verbal phrase
    else:
        trailer = "whether the student understands the concept of"  # +noun phrase

    all_concepts = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting concepts"):
        prompts = []
        for q in batch:
            q_type = q.q_type.lower().replace(SPACE, "-")
            prompt = (
                f"{q.header(1)}\n{str(q)}\n\n"
                f"Remark:\nThe above exercise is a {q_type} question that tests {trailer}"
            )
            prompts.append(prompt)

        concepts = llm.complete_prompts(prompts, stop_tokens=[".", ","], **kwargs)
        all_concepts.extend([c.strip().rstrip(".,") for c in concepts])

    return all_concepts


@torch.inference_mode()
def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int, **kwargs) -> torch.Tensor:
    all_embeddings = []
    for batch in tqdm(batched(questions, batch_size), desc="Extracting question embeddings"):
        contexts = [f"{q.header(2)}\n" for q in batch]
        texts = [str(q) for q in batch]
        all_embeddings.append(llm.encode(texts, contexts, **kwargs))

    return torch.cat(all_embeddings, dim=0)


def build_res_df(questions: list[Question], concepts: list[str]) -> pd.DataFrame:
    q_dicts = []
    for q, c in zip(questions, concepts):
        q_dict = q.flat_dict
        q_dict.pop("images", None)
        q_dict["KC"] = c
        q_dicts.append(q_dict)

    return pd.DataFrame.from_records(q_dicts)
