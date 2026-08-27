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


def test_pairing_changes_the_contrasts_not_the_summaries(run_dir):
    """The point of pairing: same fits, same folds, same summary columns — new evidence."""
    pytest.importorskip("leapfit")
    args = ["fit", "--run_dir", str(run_dir), "--folds", "2", "--seeds", "2",
            "--scheme", "item_blocked"]
    cli_main(args)
    paired = pd.read_csv(os.path.join(fit_dir(str(run_dir), "afm"), "model-comparison.csv"))
    cli_main([*args, "--unpaired"])
    unpaired = pd.read_csv(os.path.join(fit_dir(str(run_dir), "afm"), "model-comparison.csv"))

    for column in ("cv_rmse_item_blocked", "cv_rmse_sd_item_blocked",
                   "cv_unseen_item_blocked", "cv_converged_item_blocked"):
        pd.testing.assert_series_equal(paired[column], unpaired[column])
    assert not os.path.exists(os.path.join(fit_dir(str(run_dir), "afm"),
                                           "paired-contrasts.csv"))


def test_a_single_kc_model_has_no_baseline_to_contrast_against(run_dir):
    pytest.importorskip("leapfit")
    cli_main(["fit", "--run_dir", str(run_dir), "--kc_model", "fine",
              "--folds", "2", "--seeds", "1", "--scheme", "item_blocked"])
    outdir = fit_dir(str(run_dir), "afm")
    assert pd.read_csv(os.path.join(outdir, "model-comparison.csv"))["kc_model"].tolist() == ["fine"]
    assert not os.path.exists(os.path.join(outdir, "paired-contrasts.csv"))


def test_a_base_install_is_told_what_to_install(run_dir, monkeypatch):
    """The hint is only worth having if it beats ModuleNotFoundError to it."""
    monkeypatch.setitem(sys.modules, "leapfit", None)
    with pytest.raises(SystemExit, match=r'pip install "kcluster\[fit\]"'):
        cli_main(["fit", "--run_dir", str(run_dir)])


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
    assert len(folds) == 2 * 2 * 2 * 2, "two KC models x two schemes x two seeds x two folds"
    # every model scored on the same partitions, which is what makes it paired
    assert (folds.groupby(["scheme", "seed", "fold"])["kc_model"].nunique() == 2).all()

    contrasts = pd.read_csv(os.path.join(outdir, "paired-contrasts.csv"))
    assert set(contrasts["baseline"]) == {"coarse"}, "no Single-KC here, so the first by name"
    assert set(contrasts["kc_model"]) == {"fine"}
    assert {"mean_diff", "sd_diff", "folds_better"} <= set(contrasts.columns)

    aliased = pd.read_csv(os.path.join(outdir, "identification.csv"))
    assert aliased["reason"].str.contains("reference level").any()

    assert sorted(os.listdir(os.path.join(outdir, "kc-values"))) == \
        ["coarse.csv", "fine.csv"]
    predictions = pd.read_csv(os.path.join(outdir, "student-step-with-prediction.txt"),
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


# --------------------------------------------------------------------------
# One result dir per scope
# --------------------------------------------------------------------------

def _scoped_dir(tmp_path, cohorts=2, **names):
    """A run dir holding a tagged file whose cohorts are named courses."""
    frame = _tagged_frame(cohorts=cohorts)
    cohort = frame["Anon Student Id"].str.extract(r"^c(\d+)s")[0]
    frame.insert(1, "Problem Hierarchy", "Course " + cohort.map(lambda c: names.get(c, f"C{c}")))
    path = tmp_path / "ds_student-step-tagged.txt"
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return tmp_path


def test_each_scope_writes_its_own_result_dir(tmp_path):
    pytest.importorskip("leapfit")
    run = _scoped_dir(tmp_path)
    cli_main(["fit", "--run_dir", str(run), "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])

    # a fit dir per course, and none at the root; with no dir to reuse, the
    # scope keeps its own name rather than having its level type guessed away
    assert not os.path.exists(os.path.join(run, "fit"))
    for course in ("Course-C0", "Course-C1"):
        comparison = pd.read_csv(os.path.join(fit_dir(os.path.join(run, course), "afm"),
                                              "model-comparison.csv"))
        assert comparison["kc_model"].tolist() == ["coarse", "fine"]
        # one course is one block, so nothing to warn about across components
        assert not os.path.exists(os.path.join(fit_dir(os.path.join(run, course), "afm"),
                                               "kc-components.csv"))


def test_the_root_holds_only_the_two_cross_scope_files(tmp_path):
    pytest.importorskip("leapfit")
    run = _scoped_dir(tmp_path)
    cli_main(["fit", "--run_dir", str(run), "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])

    by_scope = pd.read_csv(run / "ds_afm-model-comparison-by-scope.csv")
    assert by_scope.columns[0] == "scope"
    assert sorted(by_scope["scope"].unique()) == ["Course C0", "Course C1"]
    assert len(by_scope) == 4, "two KC models x two scopes"

    predictions = pd.read_csv(run / "ds_afm-student-step-with-prediction.txt", sep="\t")
    source = pd.read_csv(run / "ds_student-step-tagged.txt", sep="\t")
    assert len(predictions) == len(source), "every scope's rows, in the file's order"
    assert predictions["Anon Student Id"].tolist() == source["Anon Student Id"].tolist()
    assert {"Predicted Error Rate (coarse)",
            "Predicted Error Rate (fine)"} <= set(predictions.columns)
    # the 200 MB-scale table is written once, not once per scope
    for course in ("Course-C0", "Course-C1"):
        assert not os.path.exists(os.path.join(fit_dir(os.path.join(run, course), "afm"),
                                               "student-step-with-prediction.txt"))


def test_a_scope_reuses_the_result_dir_its_kc_models_came_from(tmp_path):
    """``"Course C0"`` writes into an existing ``C0/``, not a new ``Course-C0/``."""
    pytest.importorskip("leapfit")
    run = _scoped_dir(tmp_path)
    os.makedirs(run / "C0" / "kc")
    cli_main(["fit", "--run_dir", str(run), "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])

    assert os.path.exists(os.path.join(fit_dir(str(run / "C0"), "afm"), "model-comparison.csv"))
    assert not os.path.exists(run / "Course-C0")
    # C1 has no dir to reuse, so it keeps the scope's own name untouched
    assert os.path.exists(os.path.join(fit_dir(str(run / "Course-C1"), "afm"),
                                       "model-comparison.csv"))


def test_no_scope_fits_the_export_as_one_model(tmp_path):
    pytest.importorskip("leapfit")
    run = _scoped_dir(tmp_path)
    cli_main(["fit", "--run_dir", str(run), "--no_scope", "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])

    outdir = fit_dir(str(run), "afm")
    assert pd.read_csv(os.path.join(outdir, "model-comparison.csv"))["kc_model"].tolist() == \
        ["coarse", "fine"]
    assert os.path.exists(os.path.join(outdir, "student-step-with-prediction.txt"))
    assert not list(run.glob("*model-comparison-by-scope.csv"))
    assert not os.path.exists(run / "C0")


def test_an_unscoped_export_is_unchanged(run_dir):
    """No hierarchy column: the run dir is the one scope's result dir."""
    pytest.importorskip("leapfit")
    cli_main(["fit", "--run_dir", str(run_dir), "--folds", "2", "--seeds", "1",
              "--scheme", "item_blocked"])
    assert os.path.exists(os.path.join(fit_dir(str(run_dir), "afm"), "model-comparison.csv"))
    assert not list(run_dir.glob("*model-comparison-by-scope.csv"))


def test_scopes_colliding_on_one_dir_name_raise(tmp_path):
    pytest.importorskip("leapfit")  # KC-model discovery runs first and goes through Leapfit
    run = _scoped_dir(tmp_path, **{"0": "A B", "1": "A-B"})
    with pytest.raises(SystemExit, match="same result dir name"):
        cli_main(["fit", "--run_dir", str(run), "--folds", "2", "--seeds", "1"])


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
