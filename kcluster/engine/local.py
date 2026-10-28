"""Local (in-process) execution substrate: a HuggingFace causal-LM wrapper and
the Lightning batch scorer that shards large scoring jobs across GPUs.

The engine is general: it knows nothing of questions or of KCluster's prompts.
What it takes is a *prompt*, the text a model reads, in either of two forms:

- a string, read verbatim. The tokenizer's own leading special tokens (a
  ``<bos>``) are added as its default says, unless the string already begins
  with the BOS token, so a prompt rendered from a chat template, which carries
  its own, is never given a second one. ``add_bos`` overrides the default
  either way: a base model whose tokenizer adds none (Gemma) wants ``True``.
- a conversation, as a list of ``{"role", "content"}`` messages, written in
  the model's chat template with the model's turn opened at the end. A last
  message in the assistant's role is a *prefill*: it is written after the
  opened turn, as the words the model continues.

The scoring methods take a prompt as the context of each *text*, the string
whose tokens are scored (``log_prob``) or pooled (``encode``). Context and
text are tokenized together, as the model reads them, and the text's tokens
are found by their character offsets, so the split is the engine's and owes
nothing to a tokenizer's pair template (which may insert a BOS between the
two, as Llama's does). A text must start at a token boundary: give it the
leading space or line break it needs, as a completion would have.
"""

import itertools
import os
from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

#: What a model reads before a text: the text itself, or a conversation as messages.
Prompt = str | list[dict]


def render_prompt(tokenizer, prompt: Prompt) -> str:
    """The text a model reads for a prompt: a string as it is, messages in the tokenizer's chat template with the
    model's turn opened, and a last assistant message written after it as a prefill."""
    if isinstance(prompt, str):
        return prompt
    messages = list(prompt)
    prefill = messages.pop()["content"] if messages and messages[-1].get("role") == "assistant" else ""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill


def _adds_bos(tokenizer) -> bool:
    """Whether the tokenizer puts its BOS token before a text on its own."""
    bos_id = tokenizer.bos_token_id
    return bos_id is not None and tokenizer("x")["input_ids"][:1] == [bos_id]


def _with_specials(tokenizer, text: str, add_bos: bool | None) -> tuple[str, bool]:
    """The text to tokenize and whether to let the tokenizer add its special tokens, under the BOS rule above."""
    bos = tokenizer.bos_token
    if bos and text.startswith(bos):
        return text, False
    if add_bos is None:
        return text, True
    if add_bos and bos and not _adds_bos(tokenizer):
        return bos + text, True
    return text, add_bos


def _encode_pair(tokenizer, context: Prompt, text: str, add_bos: bool | None) -> tuple[list[int], list[bool]]:
    """Token ids of a context and text read together, and the mask of the text's tokens."""
    context, specials = _with_specials(tokenizer, render_prompt(tokenizer, context), add_bos)
    enc = tokenizer(context + text, add_special_tokens=specials, return_offsets_mapping=True)
    boundary = len(context)
    if any(start < boundary < end for start, end in enc["offset_mapping"]):
        raise ValueError(f"the text {text[:30]!r} does not start at a token boundary after its context: give it the "
                         "leading space or line break a completion would have")
    # Special tokens the tokenizer adds carry an empty offset, and count as context.
    is_text = [start >= boundary and end > start for start, end in enc["offset_mapping"]]
    if text and not any(is_text):
        raise ValueError(f"the text {text[:30]!r} tokenizes to nothing")
    return enc["input_ids"], is_text


def pair_inputs(tokenizer, contexts: list[Prompt], texts: list[str] | None = None,
                pad_to_multiple_of: int = None, device=None, *, add_bos: bool | None = None):
    """Model inputs for a batch of prompts, or of (context, text) pairs, and the mask of each text's tokens.

    Rows are padded on the left to one length, a multiple of ``pad_to_multiple_of`` if given. The mask covers the
    text alone, leaving out the context, the special tokens and the padding; with no texts it is all False.
    """
    texts = texts if texts is not None else [""] * len(contexts)
    if len(texts) != len(contexts):
        raise ValueError("Contexts and texts must have the equal length")
    rows = [_encode_pair(tokenizer, c, t, add_bos) for c, t in zip(contexts, texts, strict=True)]
    length = max(len(ids) for ids, _ in rows)
    if pad_to_multiple_of:
        length = -(-length // pad_to_multiple_of) * pad_to_multiple_of
    pad_id = tokenizer.pad_token_id

    def padded(values, fill):
        return [[fill] * (length - len(v)) + list(v) for v in values]

    inputs = {"input_ids": torch.tensor(padded([ids for ids, _ in rows], pad_id)),
              "attention_mask": torch.tensor(padded([[1] * len(ids) for ids, _ in rows], 0))}
    is_text = torch.tensor(padded([mask for _, mask in rows], False))
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        is_text = is_text.to(device)
    return inputs, is_text


def token_count(tokenizer, context: Prompt, text: str = "", add_bos: bool | None = None) -> int:
    """Tokens a model reads for a text after its context, as ``pair_inputs`` tokenizes them."""
    ids, _ = _encode_pair(tokenizer, context, text, add_bos)
    return len(ids)


def _token_loss(model, inputs, labels, ignore_idx: int = -100):
    """Per-token cross-entropy of each label given the tokens before it, 0 where ignored."""
    logits = model(**inputs).logits.transpose(-1, -2)  # (N, V, S)
    return F.cross_entropy(logits[..., :-1], labels[:, 1:], reduction="none", ignore_index=ignore_idx)  # (N, S)


def _check_preceded(inputs, is_text) -> None:
    """Raise if any text starts the sequence: nothing predicts its first token, so it cannot be scored."""
    first = inputs["attention_mask"].int().argmax(dim=1)  # the first real token of each left-padded row
    if is_text[torch.arange(len(first)), first].any():
        raise ValueError("a text cannot be scored with nothing before it: give it a context, or a tokenizer that "
                         "adds a BOS token (add_bos=True)")


class LargeLangModel:
    """Encapsulates a downloaded LLM and its tokenizer.

    ``add_bos`` is the engine's BOS rule for string prompts (module docstring): None follows the tokenizer's default,
    True puts the model's BOS token before every prompt that lacks one, False adds no special tokens.
    """

    def __init__(self, model_path: str, device: str = "auto", add_bos: bool | None = None, **model_args):
        # Load tokenizer
        self.tokenizer = LargeLangModel.load_tokenizer(model_path)
        self.add_bos = add_bos

        # Load model
        self.model = LargeLangModel.load_model(model_path, device_map=device, **model_args).eval()

    @staticmethod
    def load_tokenizer(path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left", **kwargs)
        if not tokenizer.is_fast:
            raise ValueError(f"{path} has no fast tokenizer; the engine splits a text from its context by "
                             "character offsets, which only a fast tokenizer reports")
        if tokenizer.pad_token is None:
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

    def _inputs(self, contexts, texts=None, pad_to_multiple_of=None):
        return pair_inputs(self.tokenizer, contexts, texts, pad_to_multiple_of, self.device, add_bos=self.add_bos)

    def next_logits(self, prompts: list[Prompt], normalize: bool = False, pad_to_multiple_of: int = None):
        """Returns the next-token logits for a batch of prompts"""
        inputs, _ = self._inputs(prompts, pad_to_multiple_of=pad_to_multiple_of)
        logits = self.model(**inputs).logits[:, -1]  # (N, V)
        return torch.log_softmax(logits, dim=-1) if normalize else logits

    def next_tokens(self, prompts: list[Prompt], choices: list[str] = None,
                    top_k: int = 1, pad_to_multiple_of: int = None) -> list[tuple[str, ...]]:
        """Returns the most likely next tokens (optionally restricted) for a batch of prompts"""
        assert (not choices) or (top_k <= len(choices)), \
            f"{len(choices)} choices are insufficient for Top-{top_k} selection"

        logits = self.next_logits(prompts, pad_to_multiple_of=pad_to_multiple_of)

        # if only some tokens are allowed, mask out the other logits
        if choices:
            chc_ids = list(itertools.chain.from_iterable(self.tokenizer(choices, add_special_tokens=False)["input_ids"]))
            mask = torch.ones_like(logits[0], dtype=torch.bool)
            mask[chc_ids] = 0
            logits = torch.masked_fill(logits, mask, -torch.inf)

        token_ids = torch.topk(logits, top_k, dim=-1).indices
        next_tokens = self.tokenizer.batch_decode(token_ids.reshape(-1, 1), skip_special_tokens=True)
        return list(itertools.batched(next_tokens, top_k))

    def complete_prompts(self, prompts: list[Prompt], stop_tokens: list[str] = None, max_new_tokens: int = 200,
                         pad_to_multiple_of: int = None, **kwargs) -> list[str]:
        """Completes a batch of prompts.

        Generation ends at the model's own end of sequence, or where the completion ends with one of the
        ``stop_tokens``, strings matched on the text as generated (``generate``'s ``stop_strings``); the stop stays
        in the completion. ``kwargs`` reach ``generate``.
        """
        stops = [*(stop_tokens or []), *(kwargs.pop("stop_strings", None) or [])]
        if stops:
            kwargs.update(stop_strings=stops, tokenizer=kwargs.get("tokenizer", self.tokenizer))
        inputs, _ = self._inputs(prompts, pad_to_multiple_of=pad_to_multiple_of)
        outputs = self.model.generate(**inputs,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      max_new_tokens=max_new_tokens, **kwargs)
        num_new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
        return self.tokenizer.batch_decode(outputs[:, -num_new_tokens:], skip_special_tokens=True)

    def log_prob(self, texts: list[str], contexts: list[Prompt] = None,
                 ignore_idx: int = -100, pad_to_multiple_of: int = None, return_ppl: bool = False):
        """Computes the log-prob of a batch of texts, each given its context.

        With no contexts a text follows the BOS token alone, on a tokenizer that adds one; a text with nothing
        before it cannot be scored and raises.
        """
        contexts = contexts if contexts is not None else [""] * len(texts)
        inputs, is_text = self._inputs(contexts, texts, pad_to_multiple_of)
        _check_preceded(inputs, is_text)
        labels = torch.masked_fill(inputs["input_ids"], ~is_text, ignore_idx)
        loss = _token_loss(self.model, inputs, labels, ignore_idx)  # (N, S)

        if return_ppl:  # calculate perplexity
            return torch.exp(torch.sum(loss, dim=-1) / torch.sum(is_text, dim=-1))  # (N,)

        return -torch.sum(loss, dim=-1)  # (N,)

    def encode(self, texts: list[str], contexts: list[Prompt] = None, pad_to_multiple_of: int = None) -> torch.Tensor:
        """Encodes a batch of texts (each after its context, if any) as mean-pooled last-layer hidden states"""
        contexts = contexts if contexts is not None else [""] * len(texts)
        inputs, mask = self._inputs(contexts, texts, pad_to_multiple_of)  # mask: (N, S)
        indices = torch.cumsum(torch.sum(mask, dim=-1), dim=-1).tolist()  # (N,)

        last_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]  # (N, S, H)

        chunks = torch.vsplit(last_states[mask], indices)[:-1]  # (Nc, H) * N
        embeddings = torch.stack([torch.mean(c, dim=0) for c in chunks], dim=0)  # (N, H)
        return embeddings


def collate_pair(batch: list[tuple[Prompt, str]], tokenizer,
                 pad_to_multiple_of: int = None, ignore_idx: int = -100, add_bos: bool | None = None):
    contexts, texts = list(zip(*batch))
    inputs, is_text = pair_inputs(tokenizer, list(contexts), list(texts), pad_to_multiple_of, add_bos=add_bos)
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
                num_workers: int = 0, pad_to_multiple_of: int = None, add_bos: bool | None = None,
                **trainer_kwargs) -> None:
    """Score a grid of (context, text) pairs on the GPUs, as shards in ``shard_dir``.

    ``pairs`` is a map-style dataset of (context, text) tuples — a
    ``ScoringGrid`` (``kcluster.core.grid``) such as ``PairQuestion`` — and
    each rank writes the log-prob of every text given its context to its own
    shard pair, which ``PointwiseMutualInfo.from_shards`` reassembles.
    ``add_bos`` is the engine's BOS rule for string contexts.
    ``trainer_kwargs`` reach ``L.Trainer``: ``accelerator`` (``"gpu"`` unless
    given), ``devices`` (every GPU unless given) and ``strategy``. Returns
    once every rank has written its shard, True in the first process: what
    comes after the scores (the reassembly, the KC models) is its to do alone.
    """
    dl = DataLoader(pairs, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers,
                    collate_fn=partial(collate_pair, tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of,
                                       add_bos=add_bos))
    trainer = L.Trainer(callbacks=[CustomWriter(output_dir=shard_dir, write_interval="epoch")], logger=False,
                        **{"accelerator": "gpu", "devices": -1, **trainer_kwargs})
    trainer.predict(model, dataloaders=dl, return_predictions=False)
    trainer.strategy.barrier()
    return trainer.is_global_zero
