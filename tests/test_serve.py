"""Offline tests for the HTTP server: routes, kwarg passthrough, error
surfacing, the tokenizer guard, and the health report. A stub engine
(conftest's ``make_stub_llm``) stands in for the model, so neither torch nor a
GPU is needed."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from kcluster.engine.serve import check_tokenizer, create_app, to_jsonable


@pytest.fixture
def stub(make_stub_llm):
    return make_stub_llm()


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


def test_check_tokenizer_accepts_a_tokenizer_the_engine_can_split_with(make_tokenizer):
    pytest.importorskip("torch")
    check_tokenizer(make_tokenizer(["alpha", "beta"]))
    check_tokenizer(make_tokenizer(["alpha", "beta"], bos="<s>"))


def test_check_tokenizer_rejects_a_tokenizer_without_offsets():
    pytest.importorskip("torch")

    class SlowTokenizer:
        bos_token = None

        def __call__(self, text, **kwargs):
            if kwargs.get("return_offsets_mapping"):
                raise NotImplementedError("return_offset_mapping is not available when using Python tokenizers")
            return {"input_ids": [1, 2]}

    with pytest.raises(RuntimeError, match="cannot split a text from its context"):
        check_tokenizer(SlowTokenizer())


def test_every_route_takes_a_conversation_as_a_prompt(client, stub):
    """A client addresses a chat model with messages; the server renders them with its own template."""
    messages = [{"role": "user", "content": "Write an exercise."}]
    rsp = client.post("/complete_prompts", json={"prompts": [messages, "plain"]})
    assert rsp.status_code == 200
    rsp = client.post("/log_prob", json={"texts": ["abc"], "contexts": [messages]})
    assert rsp.status_code == 200
    assert stub.calls[-1] == ("log_prob", {"contexts": [messages], "return_ppl": False})
    rsp = client.post("/encode", json={"texts": ["abc"], "contexts": [messages]})
    assert rsp.status_code == 200
    rsp = client.post("/next_tokens", json={"prompts": [messages], "choices": ["a"]})
    assert rsp.status_code == 200
