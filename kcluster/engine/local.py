"""Local (in-process) execution substrate: a HuggingFace causal-LM wrapper and
the Lightning batch scorer that shards large scoring jobs across GPUs.

Figures: every method takes the figures its texts show (``figures``, and
``context_figures`` for the paired methods), one list of sources per text,
matching the ``[figure]`` markers in order. A model with a processor
(``ImageAdapter``) shows them as image tokens; a model without one refuses
them, and a bank's markers reach it as literal text."""

import inspect
import itertools
import os
from functools import lru_cache, partial

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from kcluster.engine.vision import ImageAdapter

#: The precisions a command loads a model in (``--dtype``). Phi-2 ran in
#: float16; Gemma overflows in it and wants bfloat16.
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def load_adapter(llm_path: str, figure_root: str | None) -> ImageAdapter | None:
    """The model's image adapter when figures are to be shown; None otherwise, or exit for a text-only model."""
    if figure_root is None:
        return None
    adapter = ImageAdapter.load(llm_path)
    if adapter is None:
        raise SystemExit(f"{llm_path} takes no images (it has no processor), but the bank shows figures — "
                         "pass --ignore_figures to score their markers as text")
    return adapter


def _no_figures(figures, n: int) -> bool:
    return figures is None or not any(figures)


def _pair_inputs(tokenizer, contexts, texts, pad_to_multiple_of: int = None, device=None, *,
                 adapter: ImageAdapter = None, context_figures=None, figures=None):
    """Model inputs for a batch of (context, text) pairs, and the mask of each text's tokens.

    The tokenizer marks the second sequence of a pair in ``token_type_ids``;
    that marker is what lets scoring and pooling cover the text alone, leaving
    out the context and the padding. With figures, the adapter first replaces
    each side's markers by image tokens and adds the pixel inputs, and the
    mask also leaves out the image soft tokens: the text's hard tokens are
    what is scored.
    """
    n = len(texts)
    extra = {}
    if not (_no_figures(figures, n) and _no_figures(context_figures, n)):
        if adapter is None:
            raise ValueError("figures were given, but this model takes no images (it has no processor)")
        rows = [[(c, cf), (t, tf)] for c, t, cf, tf in zip(contexts, texts, context_figures or [[]] * n,
                                                           figures or [[]] * n, strict=True)]
        expanded, extra = adapter.prepare(rows)
        contexts, texts = [row[0] for row in expanded], [row[1] for row in expanded]
    inputs = tokenizer(text=contexts, text_pair=texts,
                       return_tensors="pt", return_token_type_ids=True,
                       padding=True, pad_to_multiple_of=pad_to_multiple_of)
    is_text = inputs.pop("token_type_ids").bool()
    if extra:
        inputs.update(extra)
        if (mm_ids := adapter.mm_token_type_ids(inputs["input_ids"])) is not None:
            inputs["mm_token_type_ids"] = mm_ids
        is_text &= ~adapter.soft_token_mask(inputs["input_ids"])
    if device is not None:
        inputs, is_text = inputs.to(device), is_text.to(device)
    return inputs, is_text


def _prompt_inputs(tokenizer, prompts, pad_to_multiple_of: int = None, device=None, *,
                   adapter: ImageAdapter = None, figures=None):
    """Model inputs for a batch of prompts (no pair split), with their figures shown."""
    extra = {}
    if not _no_figures(figures, len(prompts)):
        if adapter is None:
            raise ValueError("figures were given, but this model takes no images (it has no processor)")
        expanded, extra = adapter.prepare([[(p, f)] for p, f in zip(prompts, figures, strict=True)])
        prompts = [row[0] for row in expanded]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=pad_to_multiple_of)
    if extra:
        inputs.update(extra)
        if (mm_ids := adapter.mm_token_type_ids(inputs["input_ids"])) is not None:
            inputs["mm_token_type_ids"] = mm_ids
    return inputs.to(device) if device is not None else inputs


@lru_cache(maxsize=None)
def _takes_logits_to_keep(model_class) -> bool:
    return "logits_to_keep" in inspect.signature(model_class.forward).parameters


def _logits_to_keep(model, labels, ignore_idx: int) -> int:
    """How many trailing positions' logits the loss needs, or 0 for all of them.

    Scoring pads on the left, so the labelled tokens sit at the end of every
    row; only the logits from just before the first labelled column onward
    enter the loss. A vocabulary of 262k tokens (Gemma 4) makes the full
    ``(N, S, V)`` logits the largest tensor of a scoring job, so a model whose
    forward takes ``logits_to_keep`` is asked for that suffix only.
    """
    if not _takes_logits_to_keep(type(model)):
        return 0
    labelled = (labels != ignore_idx).any(dim=0)
    if not labelled.any():
        return 0
    keep = labels.size(1) - int(labelled.nonzero()[0]) + 1
    return keep if keep < labels.size(1) else 0


def _token_loss(model, inputs, labels, ignore_idx: int = -100):
    """Per-token cross-entropy of each label given the tokens before it, 0 where ignored.

    One column per position the loss needs (see ``_logits_to_keep``): the
    trailing ones, or all of them. Each row's sum is the same either way.
    """
    keep = _logits_to_keep(model, labels, ignore_idx)
    logits = model(**inputs, **({"logits_to_keep": keep} if keep else {})).logits.transpose(-1, -2)  # (N, V, K)
    kept = logits.size(-1)
    return F.cross_entropy(logits[..., :-1], labels[:, -(kept - 1):], reduction="none", ignore_index=ignore_idx)


class LargeLangModel:
    """Encapsulates a downloaded LLM and its tokenizer"""

    def __init__(self, model_path: str, device: str = "auto", **model_args):
        # Load tokenizer, and the processor of a model that takes images
        self.tokenizer = LargeLangModel.load_tokenizer(model_path)
        self.images = ImageAdapter.load(model_path)

        # Load model
        self.model = LargeLangModel.load_model(model_path, device_map=device, **model_args).eval()

    @staticmethod
    def load_tokenizer(path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left", **kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @staticmethod
    def load_model(path: str, **kwargs):
        return AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token

    def next_logits(self, prompts: list[str], normalize: bool = False, pad_to_multiple_of: int = None,
                    figures: list[list] = None):
        """Returns the next-token logits for a batch of prompts"""
        inputs = _prompt_inputs(self.tokenizer, prompts, pad_to_multiple_of, self.device,
                                adapter=self.images, figures=figures)
        logits = self.model(**inputs).logits[:, -1]  # (N, V)
        return torch.log_softmax(logits, dim=-1) if normalize else logits

    def next_tokens(self, prompts: list[str], choices: list[str] = None,
                    top_k: int = 1, pad_to_multiple_of: int = None,
                    figures: list[list] = None) -> list[tuple[str, ...]]:
        """Returns the most likely next tokens (optionally restricted) for a batch of prompts"""
        assert (not choices) or (top_k <= len(choices)), \
            f"{len(choices)} choices are insufficient for Top-{top_k} selection"

        logits = self.next_logits(prompts, pad_to_multiple_of=pad_to_multiple_of, figures=figures)

        # if only some tokens are allowed, mask out the other logits
        if choices:
            chc_ids = list(itertools.chain.from_iterable(self.tokenizer(choices)["input_ids"]))
            mask = torch.ones_like(logits[0], dtype=torch.bool)
            mask[chc_ids] = 0
            logits = torch.masked_fill(logits, mask, -torch.inf)

        token_ids = torch.topk(logits, top_k, dim=-1).indices
        next_tokens = self.tokenizer.batch_decode(token_ids.reshape(-1, 1), skip_special_tokens=True)
        return list(itertools.batched(next_tokens, top_k))

    def complete_prompts(self, prompts: list[str], stop_tokens: list[str] = None, max_new_tokens: int = 200,
                         pad_to_multiple_of: int = None, figures: list[list] = None, **kwargs) -> list[str]:
        """Completes a batch of prompts"""
        if stop_ids := stop_tokens:
            stop_ids = list(itertools.chain.from_iterable(self.tokenizer(stop_tokens)["input_ids"]))
        inputs = _prompt_inputs(self.tokenizer, prompts, pad_to_multiple_of, self.device,
                                adapter=self.images, figures=figures)
        outputs = self.model.generate(**inputs,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=stop_ids,
                                      max_new_tokens=max_new_tokens, **kwargs)
        num_new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
        return self.tokenizer.batch_decode(outputs[:, -num_new_tokens:], skip_special_tokens=True)

    def log_prob(self, texts: list[str], contexts: list[str] = None,
                 ignore_idx: int = -100, pad_to_multiple_of: int = None, return_ppl: bool = False,
                 figures: list[list] = None, context_figures: list[list] = None):
        """Computes the log-prob of a batch of texts, optionally given contexts"""
        if not contexts:
            contexts = [self.pad_token] * len(texts)
        inputs, is_text = _pair_inputs(self.tokenizer, contexts, texts, pad_to_multiple_of, self.device,
                                       adapter=self.images, context_figures=context_figures, figures=figures)
        labels = torch.masked_fill(inputs["input_ids"], ~is_text, ignore_idx)
        loss = _token_loss(self.model, inputs, labels, ignore_idx)  # (N, S)

        if return_ppl:  # calculate perplexity
            return torch.exp(torch.sum(loss, dim=-1) / torch.sum(is_text, dim=-1))  # (N,)

        return -torch.sum(loss, dim=-1)  # (N,)

    def encode(self, texts: list[str], contexts: list[str] = None, pad_to_multiple_of: int = None,
               figures: list[list] = None, context_figures: list[list] = None) -> torch.Tensor:
        """Encodes a batch of texts (with optional contexts) using mean-pooled last-layer hidden states"""
        if not contexts:
            contexts = [self.pad_token] * len(texts)
        inputs, mask = _pair_inputs(self.tokenizer, contexts, texts, pad_to_multiple_of, self.device,
                                    adapter=self.images, context_figures=context_figures, figures=figures)
        indices = torch.cumsum(torch.sum(mask, dim=-1), dim=-1).tolist()  # (N,)

        last_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]  # (N, S, H)

        chunks = torch.vsplit(last_states[mask], indices)[:-1]  # (Nc, H) * N
        embeddings = torch.stack([torch.mean(c, dim=0) for c in chunks], dim=0)  # (N, H)
        return embeddings


def collate_pair(batch: list[tuple], tokenizer, pad_to_multiple_of: int = None, ignore_idx: int = -100,
                 adapter: ImageAdapter = None):
    """A batch of ``(context, text)`` or ``(context, text, context_figures, text_figures)`` items."""
    contexts, texts, *figures = zip(*batch)
    context_figures, text_figures = figures if figures else (None, None)
    inputs, is_text = _pair_inputs(tokenizer, contexts, texts, pad_to_multiple_of,
                                   adapter=adapter, context_figures=context_figures, figures=text_figures)
    return inputs, torch.masked_fill(inputs["input_ids"], ~is_text, ignore_idx)


class LogProbScorer(L.LightningModule):
    def __init__(self, llm_path: str, **model_args):
        super().__init__()
        self.model = LargeLangModel.load_model(llm_path, device_map=self.device, **model_args)

    def forward(self, batch):
        inputs, labels = batch
        return -torch.sum(_token_loss(self.model, inputs, labels), dim=-1)  # (N,)


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in output_dir, each containing the predictions of its respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # save `batch_indices` to get the information about the data index from prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


def score_pairs(model: LogProbScorer, tokenizer, pairs, shard_dir: str, *, batch_size: int = 16,
                num_workers: int = 0, pad_to_multiple_of: int = None, adapter: ImageAdapter = None,
                **trainer_kwargs) -> None:
    """Score a grid of (context, text) pairs on the GPUs, as shards in ``shard_dir``.

    ``pairs`` is a map-style dataset of (context, text) tuples — ``PairQuestion``,
    ``QuestionLO`` — each optionally followed by the two sides' figures, and
    each rank writes the log-prob of every text given its context to its own
    shard pair, which ``PointwiseMutualInfo.from_shards`` reassembles. With
    figures, ``adapter`` (``ImageAdapter.load``) shows them to the model.
    ``trainer_kwargs`` reach ``L.Trainer``: ``devices`` (every GPU unless
    given) and ``strategy``.
    """
    dl = DataLoader(pairs, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of,
                                       adapter=adapter))
    trainer = L.Trainer(accelerator="gpu", callbacks=[CustomWriter(output_dir=shard_dir, write_interval="epoch")],
                        logger=False, **{"devices": -1, **trainer_kwargs})
    trainer.predict(model, dataloaders=dl, return_predictions=False)
