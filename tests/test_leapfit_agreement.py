"""The tagger's opportunity counts, checked against Leapfit's own recomputation.

D12 gives opportunity counting exactly one owner — the tagger writes the
counts, Leapfit verifies them — which makes this the test that keeps that
arrangement honest: `StepData.opportunity_disagreements()` recomputes the
counts from the timestamps and must find nothing to disagree with. Leapfit is
the semantic authority without KCluster depending on it to *produce* anything.

Needs the ``fit`` extra (Leapfit is not a base install) and skips without it.
The real-data tests additionally need the elearning22 export and a result dir
of KC models — git-ignored local data — and skip when either is absent, so a
bare clone stays green.
"""

from pathlib import Path

import pandas as pd
import pytest

leapfit_data = pytest.importorskip(
    "leapfit.data", reason='needs the fit extra: pip install "kcluster[fit]"')

from kcluster.io.student_step import KC_COLUMN, load_student_step  # noqa: E402
from kcluster.tasks.tag import load_kc_csv, model_name, tag_student_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The pair the pipeline produces today: the driver's minimal file, tagged with
# the KC models of a run over the same question bank. Both are git-ignored, so
# these are the paths on a machine that has run elearning22 — the newest run
# dir holding models of this export's bank is picked, since a run's KC models
# and the bank must agree. Models are matched by the ``<ds>_`` prefix: another
# bank's run (elearning22-norm, say) shares the results/elearning22-* prefix
# but not the export's steps, and tagging with it fails coverage.
DS = "elearning22-mcq"
SS_PATH = ROOT / f"datasets/elearning22/data/interim/{DS}_student-step-minimal.txt"
KC_DIRS = [d for d in sorted(ROOT.glob("results/elearning22-*/kc"), reverse=True)
           if any(d.glob(f"**/{DS}_*-kc.csv"))]
# DataShop's own export of the same steps, carrying its expert opportunity
# columns — the cross-check in test_expert_opportunities_match_datashop_export.
VALIDATION_PATH = ROOT / "datasets/elearning22/data/interim/ds5426_ss_validation.txt"


def _tag(ss: pd.DataFrame, kc_models) -> pd.DataFrame:
    return tag_student_step(ss, kc_models)


def _assert_leapfit_agrees(tagged: pd.DataFrame) -> None:
    for col in tagged.columns:
        if kc := KC_COLUMN.match(col):
            data = leapfit_data.from_frame(tagged, kc["name"])
            disagreements = data.opportunity_disagreements()
            assert len(disagreements) == 0, (
                f"{kc['name']}: {len(disagreements)} disagreement(s), first at rows "
                f"{data.source_rows[disagreements[:5]]}")


def test_synthetic_agreement():
    ss = pd.DataFrame({
        "Anon Student Id": ["A", "A", "B", "A", "A"],
        "Problem Name": ["s2", "s1", "s1", "s1", "s3"],
        "Step Name": ["s2", "s1", "s1", "s1", "s3"],
        "First Attempt": ["correct", "incorrect", "correct", "hint", "correct"],
        "First Transaction Time": ["2024-01-01 10:05:00", "2024-01-01 10:00:00", "2024-01-01 10:00:00",
                                   "2024-01-01 10:05:00", "2024-01-01 10:10:00"],
        "KC (Expert)": ["E1~~E2", "E1", "E1", "E1", ""],
    })
    kc = pd.DataFrame({"id": ["s1", "s2"], "ds-step-name": ["s1", "s2"], "KC": ["alpha", "beta"]})
    _assert_leapfit_agrees(_tag(ss, {"concept": kc}))


@pytest.fixture(scope="module")
def real_tagged() -> pd.DataFrame:
    if not (SS_PATH.exists() and KC_DIRS):
        pytest.skip("elearning22 minimal export / result dir not available")
    kc_models = {model_name(str(path)): load_kc_csv(str(path))
                 for path in sorted(KC_DIRS[0].glob(f"**/{DS}_*-kc.csv"))}
    assert kc_models, f"no KC CSVs in {KC_DIRS[0]}"
    return _tag(load_student_step(str(SS_PATH)), kc_models)


def test_real_elearning22_agreement(real_tagged):
    _assert_leapfit_agrees(real_tagged)


def test_expert_opportunities_match_datashop_export(real_tagged):
    """Cross-check against the DataShop-generated export (grain of salt: theirs).

    Exact agreement is impossible: DataShop orders steps by information this
    file does not carry (its within-timestamp tie-break, and the step *start*
    time, which can precede the first transaction). Verified on this export:
    every mismatch is a permutation *within* one student x KC trajectory —
    the same rows counted 1..n in a slightly different order — with ~18% of
    cells permuted inside equal-timestamp ties and <0.1% by near-tie
    start-time swaps. So the structure is asserted exactly and the residue is
    bounded.
    """
    if not VALIDATION_PATH.exists():
        pytest.skip("ds5426_ss_validation.txt not available")
    reference = load_student_step(str(VALIDATION_PATH))
    keys = ["Anon Student Id", "Problem Name", "Step Name", "First Transaction Time"]

    for name in ("LOs-MCQ", "LOs-new-MCQ"):
        kc_col, opp_col = f"KC ({name})", f"Opportunity ({name})"
        ours = real_tagged[keys + [kc_col, opp_col]].rename(columns={opp_col: "ours"})
        theirs = reference[keys + [opp_col]].rename(columns={opp_col: "theirs"})
        merged = ours.merge(theirs, on=keys, how="outer", indicator=True)
        assert (merged["_merge"] == "both").all(), f"{name}: row sets differ"

        merged = merged[merged[kc_col] != ""]
        counts = merged.assign(ours=merged["ours"].astype(int), theirs=merged["theirs"].astype(float).astype(int))

        # same trajectories: per student x KC, both count the same rows 1..n
        per_kc = counts.groupby(["Anon Student Id", kc_col])
        assert per_kc.apply(lambda g: sorted(g["ours"]) == sorted(g["theirs"]) == list(range(1, len(g) + 1)),
                            include_groups=False).all(), f"{name}: count multisets differ somewhere"

        # exact mismatches only where this file underdetermines the order
        mismatch = counts["ours"] != counts["theirs"]
        tied = counts.duplicated(["Anon Student Id", kc_col, "First Transaction Time"], keep=False)
        residue = (mismatch & ~tied).sum() / len(counts)
        assert residue < 0.001, f"{name}: {residue:.2%} mismatches outside timestamp ties"
