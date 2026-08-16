import pandas as pd
import pytest

from kcluster.output.evaluation import (
    compute_clustering_metrics,
    eval_datashop_kc,
    evaluate_random_kc,
)


def test_metrics_are_label_invariant_and_perfect_for_same_partition():
    scores = compute_clustering_metrics(["a", "a", "b", "b"], ["x", "x", "y", "y"])
    assert all(v == pytest.approx(1.0) for v in scores.values())
    assert len(scores) == 8


def test_refinement_is_homogeneous_but_incomplete():
    # Every predicted cluster sits inside one true class, but true classes
    # are split across predicted clusters.
    scores = compute_clustering_metrics(["a", "a", "b", "b"], ["1", "2", "3", "4"])
    assert scores["Homogeneity [0, 1]"] == pytest.approx(1.0)
    assert scores["Completeness [0, 1]"] < 1.0


def test_random_baseline_is_deterministic_and_below_perfect():
    true = ["a"] * 6 + ["b"] * 6
    pred = ["x", "y"] * 6
    first = evaluate_random_kc(true, pred, num_runs=5)
    second = evaluate_random_kc(true, pred, num_runs=5)
    assert first == second
    assert first["Adj Rand Index [-1, 1]"] < 1.0


def test_eval_datashop_kc_scores_every_model_against_the_truth():
    kc_temp = pd.DataFrame(
        {
            "Problem Hierarchy": ["u"] * 4,
            "Problem Name": ["p"] * 4,
            "Step Name": ["s1", "s2", "s3", "s4"],
            "KC (truth)": ["A", "A", "B", "B"],
            "KC (good)": ["g1", "g1", "g2", "g2"],
            "KC (partial)": ["h1", None, "h2", "h2"],
        }
    )
    res = eval_datashop_kc(kc_temp, true_kcm="truth", random_kc=True, num_runs=3)
    assert res.loc["good (2 KCs)"].eq(1.0).all()  # same partition as the truth
    assert "partial (2 KCs)" in res.index  # NaN rows masked, not crashed
    assert "good-rand (2 KCs)" in res.index  # the shuffled baseline rows
    assert res.shape[1] == 8
