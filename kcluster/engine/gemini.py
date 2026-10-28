"""Gemini engine: async batched content generation.

Wraps google-genai with the patterns batch callers share: fire a
batch of prompts concurrently (optionally ramping a per-call delay to stay
under rate limits), stamp every response with the caller's id, capture
errors as data instead of exceptions, and parse answers back out — either
from plain text or from the log-probabilities of a choice token (the
prefilled-turn trick: end ``contents`` with a model-role turn like
"The answer is **" and read the next token's candidates). Dataset drivers
use it to parse problems and check answerability; genval's LAK 2026 study
scripts use it as generator and judge.

A request's ``contents`` are google-genai's own, or a *prompt* as the local
engine takes one (``kcluster.engine.local``): a string, which is the user's
one turn here, or a conversation as ``{"role", "content"}`` messages, which
``as_contents`` writes as Gemini's turns — the assistant's as the model's, a
last assistant message as the model's turn to continue, and a first system
message as the request's system instruction. So the prompts a project
writes once address a local model and Gemini alike.

No project identifiers are baked in: pass ``api_key`` (or set
``GOOGLE_API_KEY``) for the Developer API, or ``vertexai=True`` with your
own project/location.

Requires google-genai (the ``gemini`` extra).
"""

import asyncio
import heapq
import json
from collections.abc import Collection, Mapping

from google import genai
from google.genai import types
from tqdm.asyncio import tqdm as tqdm_async

#: What a model reads (``kcluster.engine``): text, or a conversation as ``{"role", "content"}`` messages.
Prompt = str | list[dict]

_ROLES = {"user": "user", "assistant": "model"}


def _is_conversation(contents) -> bool:
    """Whether contents are a conversation as messages (google-genai's content dicts carry ``parts``, not ``content``)."""
    return isinstance(contents, list) and bool(contents) and all(isinstance(m, Mapping) and "content" in m
                                                                   for m in contents)


def as_contents(prompt: Prompt) -> tuple[str | list[types.Content], str | None]:
    """Gemini's reading of a prompt: the contents, and the system instruction if the conversation opens with one.

    A string is the user's one turn. Messages are turns, the assistant's as the model's; a last assistant message
    is the model's turn to continue, as the local engine's prefill (the prefilled-turn trick above). A first message
    in the system role is the system instruction, which Gemini takes in the request's config, not among the turns.
    """
    if isinstance(prompt, str):
        return prompt, None
    messages = list(prompt)
    system = messages.pop(0)["content"] if messages and messages[0].get("role") == "system" else None
    contents = []
    for message in messages:
        if message.get("role") not in _ROLES:
            raise ValueError(f"a message's role is {message.get('role')!r}, not user or assistant (a system message "
                             "opens the conversation)")
        contents.append(types.Content(role=_ROLES[message["role"]], parts=[types.Part(text=message["content"])]))
    return contents, system


def _request(contents, config):
    """The google-genai contents and config of a request: as given, or written from a conversation (``as_contents``),
    whose system message goes in the config."""
    if not _is_conversation(contents):
        return contents, config
    contents, system = as_contents(contents)
    if system is not None:
        config = types.GenerateContentConfig.model_validate(config or {})
        if config.system_instruction is not None:
            raise ValueError("the conversation opens with a system message and the config has a system_instruction: "
                             "give one or the other")
        config = config.model_copy(update={"system_instruction": system})
    return contents, config


class GeminiEngine:
    def __init__(self, model: str, api_key: str | None = None,
                 vertexai: bool = False, project: str | None = None, location: str = "us-central1"):
        # project/location are Vertex-only: passing them on the Developer API
        # path raises "Gemini API does not support project/location."
        target = {"project": project, "location": location} if vertexai else {"api_key": api_key}
        self.client = genai.Client(vertexai=vertexai, **target)
        self.model = model
        self._next_delay = 0.0

    async def get_response(self, contents: list | Prompt, config, rsp_id: str, delay: float = None) -> str:
        """Generate content for one prompt; the result is an id-stamped JSON line.

        ``contents`` are google-genai's, or a prompt (module docstring).
        Errors come back as ``{"id": ..., "error": ...}`` instead of raising,
        so one failed call cannot sink a gathered batch. With ``delay``, calls
        are spread out by ramping each call's start time.
        """
        contents, config = _request(contents, config)  # a malformed prompt is the caller's error, and raises
        if delay is not None:
            self._next_delay += delay
            await asyncio.sleep(self._next_delay)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model, contents=contents, config=config)
            d = {"id": rsp_id} | response.to_json_dict()  # Merge the id into the response dict
            return json.dumps(d)
        except Exception as e:
            print(f"Error processing prompt {rsp_id}: {e}")
            return json.dumps({"id": rsp_id, "error": str(e)})

    async def gather_responses(self, jobs: list[tuple[list | Prompt, object, str]],
                               delay: float = None, desc: str = "Gathering responses") -> list[str]:
        """Run ``get_response`` for (contents, config, rsp_id) jobs concurrently."""
        self._next_delay = 0.0
        tasks = [asyncio.create_task(self.get_response(contents, config, rsp_id, delay=delay))
                 for contents, config, rsp_id in jobs]
        return await tqdm_async.gather(*tasks, desc=desc)


def parse_text_choices(responses: list[str], choices: Collection[str]) -> list[dict]:
    """Parse answers given as plain response text (e.g. "b" or "Yes")."""
    results = []
    for rsp in responses:
        rsp = json.loads(rsp)
        ans = rsp["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        assert ans in choices, f"Invalid answer '{ans}' for question {rsp['id']}"
        results.append({"id": rsp["id"], "answer": ans})
    return results


def sorted_choices(choices: Collection, candidates: list[dict]) -> list[tuple[float, str]]:
    """Return a max-heap of (log_probability, token) for the given choices."""
    h = []
    for cand in candidates:
        if cand["token"] in choices:
            heapq.heappush(h, (-cand["log_probability"], cand["token"]))
    assert h, "No valid choices found in candidates"
    return h


def parse_logprob_choices(responses: list[str], choices: Mapping[str, str]) -> list[dict]:
    """Parse answers from the top-1 token's log-probability candidates.

    Use with ``response_logprobs`` enabled and a prefilled model turn so the
    first generated token is the choice label; ``choices`` maps label tokens
    to answer values (e.g. ``{"a": "Yes", "b": "No"}``).
    """
    results = []
    for rsp in responses:
        rsp = json.loads(rsp)
        candidates = rsp["candidates"][0]["logprobs_result"]["top_candidates"][0]["candidates"]
        ans = sorted_choices(choices, candidates)[0][1]
        results.append({"id": rsp["id"], "answer": choices[ans]})
    return results


def save_json_responses(responses: list[str], output_path) -> None:
    with open(output_path, "w") as f:
        for r in responses:
            f.write(r + "\n")
    print(f"** Saved responses to {output_path} **")
