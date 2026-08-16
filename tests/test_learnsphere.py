"""Tests for reading LearnSphere workflow results (kcluster.io.learnsphere)."""

import pytest

from kcluster.io.learnsphere import read_cv_results


def test_read_cv_results_averages_fit_metrics_and_formats_pub_table(tmp_path):
    pytest.importorskip("bs4")
    runs = [
        (100.0, 110.0, -50.0, 0.40),
        (102.0, 112.0, -51.0, 0.42),
        (104.0, 114.0, -52.0, 0.44),
    ]
    blocks = "".join(
        f"<model><name>KC (LOs-new)</name>"
        f"<AIC>{aic}</AIC><BIC>{bic}</BIC>"
        f"<log_likelihood>{ll}</log_likelihood>"
        f"<item_blocked_cv>{cv}</item_blocked_cv></model>"
        for aic, bic, ll, cv in runs
    )
    path = tmp_path / "model_values.xml"
    path.write_text(f"<models>{blocks}</models>")

    res_table, pub_table = read_cv_results(str(path), num_cv_runs=3)
    assert res_table.loc[("aic", 0), "LOs-new"] == pytest.approx(102.0)
    assert res_table.loc["item_blocked_cv"]["LOs-new"].tolist() == [0.40, 0.42, 0.44]
    assert pub_table.loc["LOs-new", "aic"] == "102.0000 (1.6330)"


def test_read_cv_results_asserts_on_wrong_run_count(tmp_path):
    pytest.importorskip("bs4")
    path = tmp_path / "model_values.xml"
    path.write_text("<models><model><name>KC (m)</name><AIC>1.0</AIC></model></models>")
    with pytest.raises(AssertionError, match="Expected 10 results"):
        read_cv_results(str(path))
