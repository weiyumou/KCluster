"""Offline end-to-end test of the build-datashop-kc command."""

import argparse

import pandas as pd

from kcluster.commands.build_datashop_kc import main


def _write_tsv(df: pd.DataFrame, path) -> str:
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def test_build_datashop_kc_end_to_end(tmp_path):
    # Template: four steps; the expert model does not cover s4
    template = _write_tsv(
        pd.DataFrame(
            {
                "Problem Hierarchy": ["U"] * 4,
                "Problem Name": ["P"] * 4,
                "Step Name": ["s1", "s2", "s3", "s4"],
                "KC (expert)": ["e1", "e2", "e3", None],
            }
        ),
        tmp_path / "template.txt",
    )
    # The system unique-step model excludes s3
    step_kc = _write_tsv(
        pd.DataFrame(
            {
                "Problem Hierarchy": ["U"] * 4,
                "Problem Name": ["P"] * 4,
                "Step Name": ["s1", "s2", "s3", "s4"],
                "KC (Unique-step)": ["KC-1", "KC-2", None, "KC-4"],
            }
        ),
        tmp_path / "unique-step.txt",
    )
    # A KCluster result to fold in (s1 and s2 come from one merged question);
    # the D10 dataset prefix must not leak into the DataShop model name
    kc_dir = tmp_path / "kc"
    kc_dir.mkdir()
    pd.DataFrame({"ds-step-name": ["s1~s2", "s3", "s4"], "KC": ["alpha", "beta", "gamma"]}).to_csv(
        kc_dir / "questions_kcluster-unnorm-kc.csv", index=False
    )
    ss_path = _write_tsv(
        pd.DataFrame(
            {
                "Anon Student Id": ["ST1"] * 4,
                "Problem Hierarchy": ["U"] * 4,
                "Problem Name": ["P"] * 4,
                "Step Name": ["s1", "s2", "s3", "s4"],
                "First Transaction Time": ["t1", "t2", "t3", "t4"],
                "First Attempt": ["correct"] * 4,
            }
        ),
        tmp_path / "ss.txt",
    )

    main(argparse.Namespace(kc_dir=str(kc_dir), kc_temp=template, step_kc=step_kc,
                            ss_path=ss_path, multiplier=2))

    [out_dir] = [p for p in kc_dir.iterdir() if p.is_dir()]
    all_kc = pd.read_csv(out_dir / "all-kc.txt", sep="\t")
    # s1/s2 map through the tilde-joined step list; s3 is masked by the
    # unique-step model; s4 is masked because the expert model skips it.
    assert all_kc["KC (kcluster-unnorm)"].tolist()[:2] == ["alpha", "alpha"]
    assert all_kc["KC (kcluster-unnorm)"].isna().tolist() == [False, False, True, True]
    assert "KC (expert)" in all_kc.columns  # drop_other_kc=False keeps it

    merged = pd.read_csv(out_dir / "ss-merged-minimal=False-multiplier=2.txt", sep="\t")
    assert merged["Opportunity (kcluster-unnorm)"].tolist()[:2] == [1, 2]  # alpha practiced twice
    assert "KC (kcluster-unnorm-1)" in merged.columns  # the CV replica
