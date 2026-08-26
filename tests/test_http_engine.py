"""End-to-end tests for the HTTP client against a live server on a local
port, including the tasks running unmodified over the wire."""

import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from test_serve import StubLLM  # noqa: E402

from kcluster.engine.http import HttpLangModel  # noqa: E402
from kcluster.engine.serve import create_app  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    stub = StubLLM()
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
    """extract_concepts / extract_question_embeds / sort_questions only use the engine surface."""
    torch = pytest.importorskip("torch")
    from kcluster.core.question import Question
    from kcluster.tasks.concept import extract_concepts, extract_question_embeds
    from kcluster.tasks.qgen.validate import sort_questions

    url, _ = server
    llm = HttpLangModel(url)  # auto → torch
    assert llm.device == torch.device("cpu")
    qs = [Question({"id": f"q{i}", "type": "Multiple Choice",
                    "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"},
                                                                    {"label": "b", "text": "y"}]},
                    "answerKey": "a"}) for i in range(3)]
    concepts = extract_concepts(llm, qs, batch_size=2)
    assert len(concepts) == 3 and all(c.endswith("done") for c in concepts)
    embeds = extract_question_embeds(llm, qs, batch_size=2)
    assert isinstance(embeds, torch.Tensor) and embeds.shape == (3, 2)

    prompts = [f"seed {i}\n1. Stem {i}?" for i in range(3)]
    sorted_qs, sorted_prompts = sort_questions(llm, {"lo": list(qs)}, prompts, batch_size=2)
    assert len(sorted_qs["lo"]) == 3 and len(sorted_prompts) == 3  # prompts come back flattened
