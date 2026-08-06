import numpy as np
import pandas as pd
import pytest

from kcluster.core.question import Question
from kcluster.tasks.cluster import create_kc, run_ap, sim_from_embeddings


def _question(i: int) -> Question:
    return Question(
        {
            "id": f"q-{i}",
            "type": "Multiple Choice",
            "question": {"stem": f"Stem {i}?", "choices": [{"label": "a", "text": "x"}]},
            "answerKey": "a",
            "images": [f"{i}.png"],
        }
    )


@pytest.fixture
def two_blobs():
    # Six questions whose embeddings form two well-separated groups.
    questions = [_question(i) for i in range(6)]
    embeds = np.array(
        [[0.0, 0.1], [0.1, 0.0], [0.0, -0.1],
         [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]]
    )
    return questions, sim_from_embeddings(embeds, metric="euclidean")


def test_sim_from_embeddings_is_negated_distance():
    embeds = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    sim = sim_from_embeddings(embeds, metric="cosine")
    assert sim[0, 1] == pytest.approx(0.0)   # identical vectors
    assert sim[0, 2] == pytest.approx(-2.0)  # opposite vectors


def test_run_ap_recovers_the_two_groups(two_blobs):
    questions, sim = two_blobs
    kc = run_ap(questions, sim)
    assert kc.shape[0] == 6
    assert kc["KC"].str.fullmatch(r"KC-\d+").all()
    assert kc["KC"].nunique() == 2
    # Each blob lands in one cluster.
    assert kc["KC"].iloc[:3].nunique() == 1
    assert kc["KC"].iloc[3:].nunique() == 1
    assert "images" not in kc.columns


def test_run_ap_predicate_filters_questions(two_blobs):
    questions, sim = two_blobs
    kc = run_ap(questions, sim, predicate=lambda q: q["id"] in {"q-0", "q-1", "q-2"})
    assert kc["id"].tolist() == ["q-0", "q-1", "q-2"]


def test_run_ap_rejects_unknown_preference(two_blobs):
    questions, sim = two_blobs
    with pytest.raises(AssertionError):
        run_ap(questions, sim, use_p="mode")


def test_create_kc_relabels_clusters_with_exemplar_concepts(two_blobs):
    questions, sim = two_blobs
    concept_df = pd.DataFrame({"KC": ["alpha"] * 3 + ["beta"] * 3})
    kc = create_kc(concept_df, questions, sim)
    assert kc is not None
    assert "KC-raw" in kc.columns
    assert set(kc["KC"].iloc[:3]) == {"alpha"}
    assert set(kc["KC"].iloc[3:]) == {"beta"}


def test_create_kc_returns_none_when_ap_never_converges(two_blobs):
    questions, sim = two_blobs
    concept_df = pd.DataFrame({"KC": ["alpha"] * 6})
    # max_iter below convergence_iter can never satisfy the convergence check.
    assert create_kc(concept_df, questions, sim, max_iter=2) is None
