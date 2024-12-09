import itertools

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Only available for Python >= 3.12
def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


class LargeLangModel:
    """Encapsulates a downloaded LLM and its tokenizer"""

    def __init__(self, model_path: str, device: str = "auto", **model_args):
        # Load tokenizer
        self.tokenizer = LargeLangModel.load_tokenizer(model_path)

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

    def next_logits(self, prompts: list[str], normalize: bool = False, pad_to_multiple_of: int = None):
        """Returns the next-token logits for a batch of prompts"""
        inputs = self.tokenizer(prompts, return_tensors="pt",
                                padding=True, pad_to_multiple_of=pad_to_multiple_of).to(self.device)
        logits = self.model(**inputs).logits[:, -1]  # (N, V)
        return torch.log_softmax(logits, dim=-1) if normalize else logits

    def next_tokens(self, prompts: list[str], choices: list[str] = None,
                    top_k: int = 1, pad_to_multiple_of: int = None) -> list[tuple[str, ...]]:
        """Returns the most likely next tokens (optionally restricted) for a batch of prompts"""
        assert (not choices) or (top_k <= len(choices)), \
            f"{len(choices)} choices are insufficient for Top-{top_k} selection"

        logits = self.next_logits(prompts, pad_to_multiple_of=pad_to_multiple_of)

        # if only some tokens are allowed, mask out the other logits
        if choices:
            chc_ids = list(itertools.chain.from_iterable(self.tokenizer(choices)["input_ids"]))
            mask = torch.ones_like(logits[0], dtype=torch.bool)
            mask[chc_ids] = 0
            logits = torch.masked_fill(logits, mask, -torch.inf)

        token_ids = torch.topk(logits, top_k, dim=-1).indices
        next_tokens = self.tokenizer.batch_decode(token_ids.reshape(-1, 1), skip_special_tokens=True)
        return list(batched(next_tokens, top_k))

    def complete_prompts(self, prompts: list[str], stop_tokens: list[str] = None, max_new_tokens: int = 200,
                         pad_to_multiple_of: int = None, **kwargs) -> list[str]:
        """Completes a batch of prompts"""
        if stop_ids := stop_tokens:
            stop_ids = list(itertools.chain.from_iterable(self.tokenizer(stop_tokens)["input_ids"]))
        inputs = self.tokenizer(prompts, return_tensors="pt",
                                padding=True, pad_to_multiple_of=pad_to_multiple_of).to(self.device)
        outputs = self.model.generate(**inputs,
                                      eos_token_id=stop_ids,
                                      max_new_tokens=max_new_tokens, **kwargs)
        num_new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
        return self.tokenizer.batch_decode(outputs[:, -num_new_tokens:], skip_special_tokens=True)

    def log_prob(self, texts: list[str], contexts: list[str] = None,
                 ignore_idx: int = -100, pad_to_multiple_of: int = None):
        """Computes the log-prob of a batch of texts, optionally given contexts"""
        if not contexts:
            contexts = [self.pad_token] * len(texts)
        inputs = self.tokenizer(text=contexts, text_pair=texts,
                                return_tensors="pt", return_token_type_ids=True,
                                padding=True, pad_to_multiple_of=pad_to_multiple_of).to(self.device)
        mask = torch.logical_not(inputs.pop("token_type_ids"))
        labels = torch.masked_fill(inputs["input_ids"], mask, ignore_idx)

        logits = self.model(**inputs).logits.transpose(-1, -2)  # (N, V, S)
        loss = F.cross_entropy(logits[..., :-1], labels[:, 1:], reduction="none", ignore_index=ignore_idx)  # (N, S)

        return -torch.sum(loss, dim=-1)  # (N,)

    def encode(self, texts: list[str], contexts: list[str] = None, pad_to_multiple_of: int = None) -> torch.Tensor:
        """Encodes a batch of texts (with optional contexts) using mean-pooled last-layer hidden states"""
        if not contexts:
            contexts = [self.pad_token] * len(texts)
        inputs = self.tokenizer(text=contexts, text_pair=texts,
                                return_tensors="pt", return_token_type_ids=True,
                                padding=True, pad_to_multiple_of=pad_to_multiple_of).to(self.device)
        mask = inputs.pop("token_type_ids").bool()  # (N, S)
        indices = torch.cumsum(torch.sum(mask, dim=-1), dim=-1).tolist()  # (N,)

        last_states = self.model(**inputs, output_hidden_states=True).hidden_states[-1]  # (N, S, H)

        chunks = torch.vsplit(last_states[mask], indices)[:-1]  # (Nc, H) * N
        embeddings = torch.stack([torch.mean(c, dim=0) for c in chunks], dim=0)  # (N, H)
        return embeddings
