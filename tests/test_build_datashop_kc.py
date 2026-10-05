"""Tests for filling a DataShop KC-model export with a run's KC models (build-datashop-kc)."""

import pandas as pd
import pytest

from kcluster.cli import main as cli_main
from kcluster.datashop.kc_model import fill_kc_template, load_kc_template

# A KC-model export as DataShop writes it: statistics columns, an expert model
# that leaves out step id4, the empty placeholder for a new model, and a
# trailing tab on the header line. Step name s1 recurs under two problems.
TEMPLATE = (
    "Step ID\tProblem Hierarchy\tProblem Name\tMax Problem View\tStep Name\tAvg Error Step Duration (sec)\t"
    "KC (expert)\tKC (new KC model name)\t\n"
    "id1\t(unit) U, (module) M\tP1\t2.0\ts1\t0.102\te1\t\n"
    "id2\t(unit) U, (module) M\tP1\t1.0\ts2\t\te2\t\n"
    "id3\t(unit) U, (module) M\tP2\t2.0\ts1\t0.705\te1\t\n"
    "id4\t(unit) U, (module) M\tP2\t1.0\ts3\t\t\t\n"
)

#: (ds-problem-name, ds-step-name) per question; the first one spans two steps
QUESTIONS = [("P1", "s1~s2"), ("P2", "s1"), ("P2", "s3")]


def _kc(questions, labels) -> pd.DataFrame:
    """A flattened KC CSV keyed on problem and step name."""
    return pd.DataFrame({"id": [f"q{i}" for i in range(len(questions))],
                         "ds-problem-name": [problem for problem, _ in questions],
                         "ds-step-name": [steps for _, steps in questions], "KC": labels})


@pytest.fixture
def template(tmp_path) -> str:
    path = tmp_path / "ds1_kcm.txt"
    path.write_text(TEMPLATE)
    return str(path)


def test_a_run_fills_the_template_for_import(tmp_path, template):
    # Two models in the D15 subfolders of a result dir
    run = tmp_path / "run"
    for kind, model, labels in [("kcluster", "kcluster-unnorm", ["alpha", "beta", "gamma"]),
                                ("concept", "concept", ['the "carrots" approach', "b", "c"])]:
        (run / "kc" / kind).mkdir(parents=True)
        _kc(QUESTIONS, labels).to_csv(run / "kc" / kind / f"ds1-mcq_{model}-kc.csv", index=False)

    cli_main(["build-datashop-kc", "--kc_temp", template, "--run_dir", str(run)])

    header, *rows = [line.split("\t") for line in (run / "ds1_kcm-import.txt").read_text().splitlines()]
    # The export's own models are left out, so importing does not overwrite them
    assert header == ["Step ID", "Problem Hierarchy", "Problem Name", "Max Problem View", "Step Name",
                      "Avg Error Step Duration (sec)", "KC (concept)", "KC (kcluster-unnorm)"]
    # Every other column goes back as exported, text for text
    assert [row[:6] for row in rows] == [line.split("\t")[:6] for line in TEMPLATE.splitlines()[1:]]
    # Each s1 takes its own problem's label; id4, which the expert model leaves
    # out, stays blank; a quoted label goes out as it is
    assert [row[6:] for row in rows] == [['the "carrots" approach', "alpha"], ['the "carrots" approach', "alpha"],
                                         ["b", "beta"], ["", ""]]


@pytest.mark.parametrize("questions, labels, match", [
    ([("P1", "s1~s2"), ("P1", "s1"), ("P2", "s1")], ["a", "b", "c"], "more than one question"),
    ([("P1", "s1~s2")], ["a"], "no question for 1 of the 3 step"),
    ([("P1", "s1~s2"), ("P2", "s1")], ["a", ""], "without a label"),
    ([("P1", "s1~s2"), ("P2", "s1")], ["a~~b", "c"], "several KCs"),
], ids=["step-claimed-twice", "step-unreached", "label-empty", "multi-kc"])
def test_a_model_that_would_not_import_as_built_raises(template, questions, labels, match):
    with pytest.raises(ValueError, match=match):
        fill_kc_template(load_kc_template(template), {"concept": _kc(questions, labels)})


def test_a_model_already_in_the_template_raises(template):
    # Importing it would overwrite the expert model in DataShop
    with pytest.raises(ValueError, match="already in the template"):
        fill_kc_template(load_kc_template(template), {"expert": _kc(QUESTIONS, ["a", "b", "c"])})


def test_a_model_with_repeated_columns_is_one_model(tmp_path):
    # A step with two KCs repeats its model's column: together they tag id1 and id2
    path = tmp_path / "kcm.txt"
    path.write_text("Step ID\tProblem Name\tStep Name\tKC (expert)\tKC (expert)\n"
                    "id1\tP1\ts1\te1\te2\nid2\tP1\ts2\te1\t\nid3\tP2\ts1\t\t\n")
    filled = fill_kc_template(load_kc_template(str(path)), {"concept": _kc(QUESTIONS[:2], ["a", "b"])})
    assert filled.columns.tolist() == ["Step ID", "Problem Name", "Step Name", "KC (concept)"]
    assert filled["KC (concept)"].tolist() == ["a", "a", ""]


def test_a_template_without_models_has_every_step_labelled(template):
    bare = load_kc_template(template).drop(columns="KC (expert)")
    filled = fill_kc_template(bare, {"concept": _kc(QUESTIONS, ["a", "b", "c"])})
    assert filled["KC (concept)"].tolist() == ["a", "a", "b", "c"]
    with pytest.raises(ValueError, match="no question for 1 of the 4 step"):
        fill_kc_template(bare, {"concept": _kc(QUESTIONS[:2], ["a", "b"])})


def test_a_file_without_step_ids_is_not_a_template(template):
    with pytest.raises(ValueError, match="not a DataShop KC-model export"):
        fill_kc_template(load_kc_template(template).drop(columns="Step ID"),
                         {"concept": _kc(QUESTIONS, ["a", "b", "c"])})


def test_without_kc_models_there_is_nothing_to_import(template, monkeypatch):
    monkeypatch.delenv("KCLUSTER_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="No KC models"):
        cli_main(["build-datashop-kc", "--kc_temp", template])


# --- a bank that reaches fewer steps than the template's models tag ---

def test_restricting_cuts_the_templates_models_down_to_the_banks_steps(tmp_path, template):
    # The bank has no question for P2's s1 (id3), which the expert model tags
    kc_path = tmp_path / "ds1-mcq_concept-kc.csv"
    _kc(QUESTIONS[:1], ["a"]).to_csv(kc_path, index=False)

    cli_main(["build-datashop-kc", "--kc_temp", template, "--kc_paths", str(kc_path), "--restrict_suffix", "MCQ"])

    header, *rows = [line.split("\t") for line in (tmp_path / "ds1_kcm-import.txt").read_text().splitlines()]
    # The expert model goes out again under the suffix, beside the new model
    # and over the same steps; the original stays out, so it is not overwritten
    assert header[6:] == ["KC (expert-MCQ)", "KC (concept)"]
    assert [row[6:] for row in rows] == [["e1", "a"], ["e2", "a"], ["", ""], ["", ""]]
    assert [row[:6] for row in rows] == [line.split("\t")[:6] for line in TEMPLATE.splitlines()[1:]]


def test_restricting_keeps_to_the_steps_every_new_model_reaches(template):
    filled = fill_kc_template(load_kc_template(template),
                              {"concept": _kc(QUESTIONS, ["a", "b", "c"]), "narrow": _kc(QUESTIONS[1:], ["x", "y"])},
                              restrict_suffix="MCQ")
    # Only id3 is tagged by the expert model and reached by both banks
    assert filled[["KC (expert-MCQ)", "KC (concept)", "KC (narrow)"]].values.tolist() == \
        [["", "", ""], ["", "", ""], ["e1", "b", "x"], ["", "", ""]]


def test_a_restricted_model_keeps_its_repeated_columns(tmp_path):
    # DataShop's form for a step with two KCs: the model's column, twice
    path = tmp_path / "kcm.txt"
    path.write_text("Step ID\tProblem Name\tStep Name\tKC (expert)\tKC (expert)\n"
                    "id1\tP1\ts1\te1\te2\nid2\tP1\ts2\te1\t\nid3\tP2\ts1\te3\te4\n")
    filled = fill_kc_template(load_kc_template(str(path)), {"concept": _kc(QUESTIONS[:1], ["a"])},
                              restrict_suffix="MCQ")
    assert filled.columns.tolist() == ["Step ID", "Problem Name", "Step Name",
                                       "KC (expert-MCQ)", "KC (expert-MCQ)", "KC (concept)"]
    assert filled.iloc[:, 3:].values.tolist() == [["e1", "e2", "a"], ["e1", "", "a"], ["", "", ""]]


@pytest.mark.parametrize("kc_models, match", [
    ({"expert-MCQ": _kc(QUESTIONS, ["a", "b", "c"])}, "named like the cut-down copies"),
    ({"concept": _kc([("P9", "s9")], ["a"])}, "no step is left to label"),
    ({}, "no KC models"),
], ids=["name-taken-by-a-copy", "nothing-shared", "no-models"])
def test_a_restriction_that_cannot_be_made_raises(template, kc_models, match):
    with pytest.raises(ValueError, match=match):
        fill_kc_template(load_kc_template(template), kc_models, restrict_suffix="MCQ")


def test_a_restricted_name_already_in_the_template_raises(template):
    # expert-MCQ exists already: writing the copy would overwrite it in DataShop
    both = load_kc_template(template)
    both["KC (expert-MCQ)"] = both["KC (expert)"]
    with pytest.raises(ValueError, match="already in the template"):
        fill_kc_template(both, {"concept": _kc(QUESTIONS, ["a", "b", "c"])}, restrict_suffix="MCQ")
