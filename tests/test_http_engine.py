"""End-to-end tests for the HTTP client against a live server on a local
port, including the tasks running unmodified over the wire."""

import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from kcluster.engine.http import HttpLangModel
from kcluster.engine.serve import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(make_stub_llm):
    stub = make_stub_llm()
    port = _free_port()
    config = uvicorn.Config(create_app(stub, model_id="stub-model"), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("server did not start")
    yield f"http://127.0.0.1:{port}", stub
    srv.should_exit = True
    thread.join(timeout=5)


def test_health_and_model_id(server):
    url, _ = server
    llm = HttpLangModel(url, tensors="list")
    assert llm.health()["model_id"] == "stub-model"
    assert llm.model_id == "stub-model"


def test_methods_mirror_the_engine_surface(server):
    url, stub = server
    llm = HttpLangModel(url, tensors="list")
    assert llm.complete_prompts(["p"], stop_tokens=["."], num_beams=3) == ["p done"]
    assert stub.calls[-1] == ("complete_prompts", {"num_beams": 3})
    assert llm.log_prob(["abcd"], contexts=["c"], return_ppl=True) == [-4.0]
    assert llm.next_tokens(["p"], choices=["a", "b"], top_k=2) == [("a", "b")]
    assert llm.next_logits(["p"]) == [[0.1, 0.2, 0.7]]
    assert llm.encode(["t"], ["c"]) == [[1.0, 2.0]]
    assert llm.tokenizer.encode("one two") == [3, 3]
    assert llm.tokenizer(["one", "two three"])["input_ids"] == [[3], [3, 5]]


def test_conversations_go_over_the_wire_as_they_are(server):
    url, stub = server
    llm = HttpLangModel(url, tensors="list")
    messages = [{"role": "user", "content": "Write an exercise."}]
    assert llm.log_prob(["abcd"], contexts=[messages]) == [-4.0]
    assert stub.calls[-1] == ("log_prob", {"contexts": [messages], "return_ppl": False})
    assert llm.complete_prompts([messages, "p"]) == ["[{'role': 'user', 'content': 'Write an exercise.'}] done", "p done"]


def test_tokenizer_shim_refuses_what_it_cannot_do(server):
    url, _ = server
    with pytest.raises(NotImplementedError, match="return_tensors"):
        HttpLangModel(url).tokenizer(["x"], return_tensors="pt")


def test_server_errors_become_runtime_errors_with_detail(server):
    url, _ = server
    with pytest.raises(RuntimeError, match="ValueError: Contexts and texts"):
        HttpLangModel(url, tensors="list").encode(["a", "b"], ["only one"])


def test_numpy_tensors(server):
    np = pytest.importorskip("numpy")
    url, _ = server
    out = HttpLangModel(url, tensors="numpy").encode(["a", "b"])
    assert isinstance(out, np.ndarray) and out.shape == (2, 2)


def test_tasks_run_unmodified_over_http(server):
    """extract_concepts / extract_question_embeds only use the engine surface."""
    torch = pytest.importorskip("torch")
    from kcluster.core.question import Question
    from kcluster.tasks.concept import extract_concepts, extract_question_embeds

    url, _ = server
    llm = HttpLangModel(url)  # auto → torch
    assert llm.device == torch.device("cpu")
    qs = [Question({"id": f"q{i}", "type": "Multiple Choice",
                    "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"},
                                                                    {"label": "b", "text": "y"}]},
                    "answerKey": "a"}) for i in range(3)]
    concepts = extract_concepts(llm, qs, batch_size=2)
    # The stub echoes the prompt, which the cleaner reads up to its first line break
    assert concepts == ["Exercise 1:"] * 3
    embeds = extract_question_embeds(llm, qs, batch_size=2)
    assert isinstance(embeds, torch.Tensor) and embeds.shape == (3, 2)
