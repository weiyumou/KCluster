"""Tests for the DataShop student-step export loader."""

import pandas as pd
import pytest

from kcluster.io.loaders.datashop_export import load_export, reduce_to_steps, universe_steps


def _write_export(path, rows, **extra) -> str:
    """A miniature export: (hierarchy, problem, step, expert KC, other KC) per row."""
    data = {
        "Anon Student Id": [f"stu{i % 2}" for i in range(len(rows))],
        "Problem Hierarchy": [h for h, _, _, _, _ in rows],
        "Problem Name": [p for _, p, _, _, _ in rows],
        "Step Name": [s for _, _, s, _, _ in rows],
        "First Transaction Time": [f"2022-08-30 13:{i:02d}:00" for i in range(len(rows))],
        "First Attempt": "correct",
        "KC (expert)": [a for _, _, _, a, _ in rows],
        "KC (other)": [b for _, _, _, _, b in rows],
        "Opportunity (expert)": 1,
        "Step Duration (sec)": 3.5,
    }
    data.update(extra)
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    return str(path)


def _rows():
    return [
        ("unit B", "P2", "s3", "e3", "o3"),
        ("unit A", "P1", "s1", "e1", "o1"),
        ("unit A", "P1", "s2", "e2", ""),      # tagged by expert only
        ("unit A", "P1", "s1", "e1", "o1"),    # repeat of an earlier step
        ("unit A", "P9", "s1", "", ""),        # same step name, different problem
    ]


def test_load_export_keeps_only_what_was_asked_for_in_order(tmp_path):
    export = load_export(_write_export(tmp_path / "e.txt", _rows()), ("expert",))
    assert list(export.columns) == ["Anon Student Id", "Problem Hierarchy", "Problem Name",
                                    "Step Name", "First Transaction Time", "First Attempt",
                                    "KC (expert)"]
    assert len(export) == 5


def test_load_export_reads_everything_as_text(tmp_path):
    # A student id that looks numeric must not come back as an int
    path = _write_export(tmp_path / "e.txt", _rows(), **{"Anon Student Id": ["0123"] * 5})
    assert load_export(path)["Anon Student Id"].tolist() == ["0123"] * 5


def test_universe_steps_keeps_steps_every_model_tags(tmp_path):
    export = load_export(_write_export(tmp_path / "e.txt", _rows()), ("expert", "other"))
    # s2 is tagged by expert but not other; the P9 copy of s1 by neither. s1
    # survives because its P1 copy is tagged by both.
    assert universe_steps(export, ("expert", "other")) == ["s1", "s3"]
    assert universe_steps(export, ("expert",)) == ["s1", "s2", "s3"]


def test_universe_steps_orders_by_the_key_not_by_step_name(tmp_path):
    # Sorted on (hierarchy, problem, step), s3 lands last despite sorting
    # before s9 alphabetically — this order becomes questions' ds-step-name.
    rows = [("unit B", "P2", "s3", "e", "o"), ("unit A", "P1", "s9", "e", "o")]
    export = load_export(_write_export(tmp_path / "e.txt", rows), ("expert", "other"))
    assert universe_steps(export, ("expert", "other")) == ["s9", "s3"]


def test_reduce_to_steps_drops_other_rows_and_keeps_export_order(tmp_path):
    export = load_export(_write_export(tmp_path / "e.txt", _rows()), ("expert",))
    reduced = reduce_to_steps(export, ["s1"])
    # both P1 rows and the untagged P9 row: filtering is by step name, and a
    # row inside the universe is kept even where the experts left it untagged
    assert reduced["Problem Name"].tolist() == ["P1", "P1", "P9"]
    assert reduced["KC (expert)"].tolist() == ["e1", "e1", ""]


def test_reduce_to_steps_requires_the_contract_columns(tmp_path):
    export = load_export(_write_export(tmp_path / "e.txt", _rows()))
    with pytest.raises(ValueError, match=r"missing required column\(s\).*First Attempt"):
        reduce_to_steps(export.drop(columns=["First Attempt"]), ["s1"])
