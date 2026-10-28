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
engine does (``kcluster.engine``): strings, or conversations as messages, and
the images the prompts place, for a model that takes them.

``figure_dir`` shows the questions' figures to such a model: each ``[figure]``
marker becomes the model's image token (``show_figures``) and the files,
resolved against that root, go beside the prompts. None, the default, shows
none: the markers reach the model as literal text.
"""

from collections.abc import Callable, Sequence
from itertools import batched

import torch
from tqdm import tqdm

from kcluster.core.prompts import CONCEPT_STOP_TOKENS, STOCK, Prompt, PromptStyle, clean_concept, show_figures
from kcluster.core.question import Question
from kcluster.engine.local import LargeLangModel
from kcluster.io.figures import figure_paths


def _shown(llm, questions: list[Question], prompts: list[Prompt], figure_dir: str | None):
    """The prompts with their figures placed for the model, and the figures to show: the ``images`` argument,
    None when no figure is shown."""
    if figure_dir is None:
        return prompts, None
    token = getattr(llm, "image_token", None)
    if not token:
        raise ValueError("this model takes no images: score the bank without figure_dir, its markers as text")
    return [show_figures(p, token) for p in prompts], [figure_paths(q, figure_dir) for q in questions]


def _batched(batch_size: int, *columns):
    """Batches of the columns zipped, each column None for one of Nones."""
    n = len(next(c for c in columns if c is not None))
    rows = zip(*[c if c is not None else [None] * n for c in columns], strict=True)
    return batched(rows, batch_size)


@torch.inference_mode()
def complete_names(llm: LargeLangModel, prompts: Sequence[Prompt], batch_size: int,
                   stop_tokens: Sequence[str] = CONCEPT_STOP_TOKENS, num_names: int | None = None,
                   ids: Sequence | None = None, desc: str = "Naming", clean: Callable[[str], str] = clean_concept,
                   images: Sequence[list] | None = None, **kwargs) -> list:
    """The name each prompt asks the model to complete, or with ``num_names`` that many names per prompt.

    A name is what ``clean`` reads out of a completion: the published reading (``clean_concept``) takes the text
    up to its first stop token or line break; a prompt style has its own (``PromptStyle.clean_concept``). ``images``
    are the images the prompts place, one list per prompt, for a model that takes them. ``kwargs`` reach the
    model's ``generate``. With ``num_names`` the model returns that many completions per prompt
    (``num_return_sequences``; beam search needs at least as many beams, sampling ``do_sample=True``), and the
    result is a list of names per prompt, the completions that name nothing (no letter) left out; a prompt left
    with no name raises, named by its entry in ``ids`` if given. A model that gives any other number of
    completions per prompt than asked for raises too, rather than pairing names with the wrong prompts.
    """
    if num_names:
        kwargs["num_return_sequences"] = num_names
    ids = ids if ids is not None else range(len(prompts))
    all_names = []
    for batch in tqdm(_batched(batch_size, prompts, ids, images), desc=desc):
        texts = [prompt for prompt, _, _ in batch]
        shown = {"images": [i for _, _, i in batch]} if images is not None else {}
        completions = llm.complete_prompts(texts, stop_tokens=list(stop_tokens), **shown, **kwargs)
        if len(completions) != len(batch) * (num_names or 1):
            raise ValueError(f"the model gave {len(completions)} completions for {len(batch)} prompts, "
                             f"not {num_names or 1} per prompt")
        names = [clean(c) for c in completions]
        if not num_names:
            all_names.extend(names)
            continue
        for (_, item, _), group in zip(batch, batched(names, num_names)):
            kept = [name for name in group if any(ch.isalpha() for ch in name)]
            if not kept:
                raise ValueError(f"the model gave no name for {item}")
            all_names.append(kept)

    return all_names


def extract_concepts(llm: LargeLangModel, questions: list[Question], batch_size: int, verbal: bool = False,
                     prompts: PromptStyle = STOCK, num_names: int | None = None, figure_dir: str | None = None,
                     **kwargs) -> list:
    """The key concept of each question, or with ``num_names`` that many names of each (``complete_names``);
    ``figure_dir`` shows the questions' figures (module docstring)."""
    shown, images = _shown(llm, questions, [prompts.concept(q, verbal=verbal) for q in questions], figure_dir)
    return complete_names(llm, shown, batch_size, stop_tokens=prompts.concept_stop_tokens, num_names=num_names,
                          ids=[f"question {q.get('id')}" for q in questions], desc="Extracting concepts",
                          clean=prompts.clean_concept, images=images, **kwargs)


@torch.inference_mode()
def embed_texts(llm: LargeLangModel, texts: Sequence[str], contexts: Sequence[Prompt], batch_size: int,
                desc: str = "Embedding", images: Sequence[list] | None = None,
                context_images: Sequence[list] | None = None, **kwargs) -> torch.Tensor:
    """The engine's embedding of each text after its context, batched: (N, H). ``images`` are the images the texts
    place and ``context_images`` those the contexts place, one list each, for a model that takes them."""
    all_embeddings = []
    for batch in tqdm(_batched(batch_size, texts, contexts, images, context_images), desc=desc):
        shown = {**({"images": [i for _, _, i, _ in batch]} if images is not None else {}),
                 **({"context_images": [i for _, _, _, i in batch]} if context_images is not None else {})}
        all_embeddings.append(llm.encode([t for t, *_ in batch], [c for _, c, *_ in batch], **shown, **kwargs))

    return torch.cat(all_embeddings, dim=0)


def extract_question_embeds(llm: LargeLangModel, questions: list[Question], batch_size: int,
                            prompts: PromptStyle = STOCK, figure_dir: str | None = None,
                            **kwargs) -> torch.Tensor:
    """Each question's text embedded after its marginal congruity context (``Question-emb`` in EDM 2025); its
    figures, with ``figure_dir``, are in the text, and pooled out of the embedding as the engine pools out images."""
    texts, images = _shown(llm, questions, [str(q) for q in questions], figure_dir)
    return embed_texts(llm, texts, [prompts.marginal(q) for q in questions], batch_size,
                       desc="Extracting question embeddings", images=images, **kwargs)
