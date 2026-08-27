"""Offline tests for the HTTP server: routes, kwarg passthrough, error
surfacing, the tokenizer guard, and the health report. A stub engine stands
in for the model, so neither torch nor a GPU is needed."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from kcluster.engine.serve import check_tokenizer, create_app, to_jsonable  # noqa: E402


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        return {"input_ids": [[len(w) for w in t.split()] for t in texts]}


class StubLLM:
    """Records calls; returns shapes matching LargeLangModel's contract (as plain lists)."""

    device = "cpu"
    tokenizer = StubTokenizer()

    def __init__(self):
        self.calls = []

    def complete_prompts(self, prompts, stop_tokens=None, max_new_tokens=200, pad_to_multiple_of=None, **kwargs):
        self.calls.append(("complete_prompts", kwargs))
        return [p + " done" for p in prompts]

    def log_prob(self, texts, contexts=None, ignore_idx=-100, pad_to_multiple_of=None, return_ppl=False):
        self.calls.append(("log_prob", {"contexts": contexts, "return_ppl": return_ppl}))
        return [-float(len(t)) for t in texts]

    def next_tokens(self, prompts, choices=None, top_k=1, pad_to_multiple_of=None):
        self.calls.append(("next_tokens", {"choices": choices, "top_k": top_k}))
        return [tuple((choices or ["x"])[:top_k]) for _ in prompts]

    def next_logits(self, prompts, normalize=False, pad_to_multiple_of=None):
        return [[0.1, 0.2, 0.7] for _ in prompts]

    def encode(self, texts, contexts=None, pad_to_multiple_of=None):
        if contexts is not None and len(contexts) != len(texts):
            raise ValueError("Contexts and texts must have the equal length")
        return [[1.0, 2.0] for _ in texts]


@pytest.fixture
def stub():
    return StubLLM()


@pytest.fixture
def client(stub):
    return TestClient(create_app(stub, model_id="stub-model"))


def test_health_reports_model_and_counters(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_id"] == "stub-model"
    assert body["device"] == "cpu"
    assert body["requests"] == 0
    client.post("/complete_prompts", json={"prompts": ["a"]})
    assert client.get("/health").json()["requests"] == 1


def test_every_response_carries_the_model_id(client):
    rsp = client.post("/log_prob", json={"texts": ["abc"]})
    assert rsp.status_code == 200
    assert rsp.json() == {"model_id": "stub-model", "result": [-3.0]}


def test_complete_forwards_unknown_fields_to_generate(client, stub):
    rsp = client.post("/complete_prompts", json={"prompts": ["p"], "stop_tokens": ["."],
                                                 "num_beams": 5, "length_penalty": -0.1})
    assert rsp.json()["result"] == ["p done"]
    _, kwargs = stub.calls[-1]
    assert kwargs == {"num_beams": 5, "length_penalty": -0.1}


def test_scoring_routes_reject_unknown_fields(client):
    rsp = client.post("/log_prob", json={"texts": ["a"], "num_beams": 5})
    assert rsp.status_code == 422


def test_next_tokens_and_tokenize_shapes(client):
    rsp = client.post("/next_tokens", json={"prompts": ["p", "q"], "choices": ["a", "b", "c"], "top_k": 2})
    assert rsp.json()["result"] == [["a", "b"], ["a", "b"]]
    rsp = client.post("/tokenize", json={"texts": ["one two", "three"]})
    assert rsp.json()["result"] == [[3, 3], [5]]


def test_engine_errors_surface_as_500_with_the_message(client):
    rsp = client.post("/encode", json={"texts": ["a", "b"], "contexts": ["only one"]})
    assert rsp.status_code == 500
    assert "ValueError: Contexts and texts must have the equal length" in rsp.json()["detail"]


def test_openapi_lists_the_engine_surface(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/complete_prompts", "/log_prob", "/next_tokens", "/next_logits", "/encode",
            "/tokenize"} <= set(paths)


def test_to_jsonable_handles_tensors_and_tuples():
    np = pytest.importorskip("numpy")
    assert to_jsonable(np.arange(3)) == [0, 1, 2]
    assert to_jsonable([("a", "b"), ("c",)]) == [["a", "b"], ["c"]]
    assert to_jsonable("text") == "text"


class PairAwareTokenizer:
    def __call__(self, text, text_pair=None, return_token_type_ids=False):
        return {"token_type_ids": [[0, 0, 1, 1]]}


class PairBlindTokenizer:
    def __call__(self, text, text_pair=None, return_token_type_ids=False):
        return {"token_type_ids": [[0, 0, 0, 0]]}


class NoTypeIdsTokenizer:
    def __call__(self, text, text_pair=None, return_token_type_ids=False):
        return {"input_ids": [[1, 2, 3, 4]]}


def test_check_tokenizer_accepts_pair_marking_tokenizers():
    check_tokenizer(PairAwareTokenizer())


@pytest.mark.parametrize("tokenizer", [PairBlindTokenizer(), NoTypeIdsTokenizer()])
def test_check_tokenizer_rejects_tokenizers_that_cannot_split_pairs(tokenizer):
    with pytest.raises(RuntimeError, match="cannot separate the context"):
        check_tokenizer(tokenizer)


def test_phi2_style_tokenizer_passes_the_guard():
    """The real check, when transformers is installed: any fast tokenizer marks pairs."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.WordLevel({"alpha": 0, "beta": 1, "[UNK]": 2}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    check_tokenizer(PreTrainedTokenizerFast(tokenizer_object=raw))
    del transformers
