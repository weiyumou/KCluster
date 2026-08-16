"""Tests for student-model fitting (kcluster.tasks.fit, kcluster.commands.fit).

The end-to-end tests need Leapfit — the optional ``fit`` extra — and skip
without it. The discovery and error paths do not, and run everywhere.
"""

import os
import sys

import pandas as pd
import pytest

from kcluster.cli import main as cli_main
from kcluster.commands.fit import _default_export
from kcluster.paths import fit_dir
from kcluster.tasks import fit as fit_task


def _tagged_frame(cohorts=1, n_students=6, n_steps=4, n_reps=3) -> pd.DataFrame:
    """A tagged student-step file with two KC models over ``cohorts`` cohorts.

    With two cohorts the students split into groups sharing no steps and no
    KCs — the disconnected case that gives each group its own intercept level.
    """
    rows = []
    for cohort in range(cohorts):
        for s in range(n_students):
            seen: dict[str, int] = {}
            for rep in range(n_reps):
                for step in range(n_steps):
                    kc = f"c{cohort}kc{step % 2}"
                    seen[kc] = seen.get(kc, 0) + 1
                    rows.append({
                        "Anon Student Id": f"c{cohort}s{s}",
                        "Problem Name": f"p{cohort}",
                        "Step Name": f"c{cohort}st{step}",
                        "First Attempt": "correct" if (s + step + rep) % 3 else "incorrect",
                        "First Transaction Time": f"2024-01-01 10:{rep:02d}:{step:02d}",
                        "KC (coarse)": f"c{cohort}",
                        "Opportunity (coarse)": str(rep * n_steps + step + 1),
                        "KC (fine)": kc,
                        "Opportunity (fine)": str(seen[kc]),
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def run_dir(tmp_path):
    """A result dir holding one tagged file, as `kcluster tag` leaves it."""
    path = tmp_path / "ds_student-step-tagged.txt"
    _tagged_frame().to_csv(path, sep="\t", index=False, lineterminator="\n")
    return tmp_path


# --------------------------------------------------------------------------
# Discovery and errors — no Leapfit needed
# --------------------------------------------------------------------------

def test_the_tagged_file_is_found_in_a_result_dir(run_dir):
    assert _default_export(str(run_dir)).endswith("ds_student-step-tagged.txt")


def test_two_tagged_files_ask_which_one(run_dir):
    (run_dir / "other_student-step-tagged.txt").write_text("")
    with pytest.raises(SystemExit, match="found 2"):
        _default_export(str(run_dir))


def test_no_run_dir_and_no_path_says_so():
    with pytest.raises(SystemExit, match="--ss_path"):
        _default_export(None)


def test_missing_leapfit_names_the_extra_to_install(monkeypatch):
    """A base install must fail with the install command, not an ImportError."""
    monkeypatch.setitem(sys.modules, "leapfit", None)
    with pytest.raises(SystemExit, match=r"kcluster\[fit\]"):
        fit_task.family_callables("afm")


def test_unknown_family_is_rejected():
    pytest.importorskip("leapfit")
    with pytest.raises(ValueError, match="bkt"):
        fit_task.family_callables("bkt")


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_fit_writes_the_family_tree(run_dir):
    pytest.importorskip("leapfit")
    cli_main(["fit", "--run_dir", str(run_dir), "--folds", "2", "--seeds", "2"])

    outdir = fit_dir(str(run_dir), "afm")
    comparison = pd.read_csv(os.path.join(outdir, "model-comparison.csv"))
    assert comparison["kc_model"].tolist() == ["coarse", "fine"]
    assert comparison["is_optimal"].all()
    assert {"cv_rmse_student_blocked", "cv_rmse_item_blocked"} <= set(comparison.columns)

    folds = pd.read_csv(os.path.join(outdir, "cv-folds.csv"))
    assert len(folds) == 2 * 2 * 2, "two KC models x two schemes x two seeds"

    aliased = pd.read_csv(os.path.join(outdir, "identification.csv"))
    assert aliased["reason"].str.contains("reference level").any()

    assert sorted(os.listdir(os.path.join(outdir, "kc-values"))) == \
        ["coarse.csv", "fine.csv"]
    predictions = pd.read_csv(os.path.join(outdir, "ds_student-step-predictions.txt"),
                              sep="\t")
    assert {"Predicted Error Rate (coarse)",
            "Predicted Error Rate (fine)"} <= set(predictions.columns)


def test_one_cohort_writes_no_component_map(run_dir):
    pytest.importorskip("leapfit")
    cli_main(["fit", "--run_dir", str(run_dir), "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])
    assert not os.path.exists(
        os.path.join(fit_dir(str(run_dir), "afm"), "kc-components.csv"))


def test_disjoint_cohorts_get_a_component_map(tmp_path):
    """Intercepts are comparable only within a cohort, so the map is a result."""
    pytest.importorskip("leapfit")
    path = tmp_path / "two_student-step-tagged.txt"
    _tagged_frame(cohorts=2).to_csv(path, sep="\t", index=False, lineterminator="\n")
    cli_main(["fit", "--run_dir", str(tmp_path), "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])

    components = pd.read_csv(os.path.join(fit_dir(str(tmp_path), "afm"),
                                          "kc-components.csv"))
    fine = components[components["kc_model"] == "fine"]
    assert set(fine["component"]) == {1, 2}
    assert fine.groupby("component")["KC Name"].nunique().tolist() == [2, 2]

    aliased = pd.read_csv(os.path.join(fit_dir(str(tmp_path), "afm"),
                                       "identification.csv"))
    reference = aliased[aliased["reason"].str.startswith("reference level")]
    assert (reference["kc_model"] == "fine").sum() == 2, "one per cohort"


def test_several_families_write_side_by_side(run_dir):
    pytest.importorskip("leapfit")
    cli_main(["fit", "--run_dir", str(run_dir), "--model", "afm", "--model", "pfa",
              "--folds", "2", "--seeds", "1", "--scheme", "item_blocked"])
    for family in ("afm", "pfa"):
        assert os.path.exists(os.path.join(fit_dir(str(run_dir), family),
                                           "model-comparison.csv"))


def test_kc_model_selection_and_its_error(run_dir):
    pytest.importorskip("leapfit")
    cli_main(["fit", "--run_dir", str(run_dir), "--kc_model", "fine",
              "--folds", "2", "--seeds", "1", "--scheme", "item_blocked"])
    comparison = pd.read_csv(os.path.join(fit_dir(str(run_dir), "afm"),
                                          "model-comparison.csv"))
    assert comparison["kc_model"].tolist() == ["fine"]

    with pytest.raises(SystemExit, match="Unknown KC model"):
        cli_main(["fit", "--run_dir", str(run_dir), "--kc_model", "nope"])
