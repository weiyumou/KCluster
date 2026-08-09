"""Does question congruity measure shared content, or shared format?

The congruity grid conditions each scored question on another question rendered
in full — type line, choice ladder, answer trailer. When the two share a format,
the scored question's scaffolding gets cheaper, and the pair scores as congruent
for reasons that have nothing to do with the knowledge component it tests. In a
corpus where every question is the same format that costs nothing; in one with a
format mix it is a confound.

This driver builds the smallest question set that can separate the two, and
scores the arms once they have been run through ``kcluster pmi``.

**Build** (``--out``). Format and CCSS code are partly confounded in Foundational
ASSIST — a third of the primary codes appear in only one format, and for those no
metric can tell content similarity from format similarity. So the probe set is
*crossed*: only codes carrying at least ``--min_*`` questions of all three formats
are kept, and a fixed quota of each is drawn. Every (code x format) cell is then
populated and both contrasts below are estimable.

**Score** (``--score``). Two AUCs per arm, each computed on the slice of pairs
where the other factor cannot interfere:

  signal  P(same code ranks above different code), over CROSS-format pairs only.
          Format composition is identical in both classes, so this is content
          discrimination with the confound held fixed.
  leak    P(same format ranks above different format), over DIFFERENT-code pairs
          only. Those pairs share no expert-assigned content, so 0.5 is the
          no-leak value and anything above it is format.

A renderer is worth adopting if leak falls toward 0.5 while signal holds. If no
arm moves leak, the prompt is not where the fix belongs.
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.jsonl import dump_questions, load_questions

# The three Foundational ASSIST formats, in the order they are reported.
FILL_IN = "Fill-in-the-blank(s)"
SELECT_ONE = "Multiple Choice (select 1)"
SELECT_ALL = "Multiple Choice (select all)"
FORMATS = (FILL_IN, SELECT_ONE, SELECT_ALL)

# Select-all is the scarcest format (227 of 2,019) and sets the quota ceiling:
# asking for more than 2 per code shrinks the design from 19 codes to 11.
DEFAULT_QUOTA = {FILL_IN: 4, SELECT_ONE: 4, SELECT_ALL: 2}


def primary_code(question) -> str | None:
    """The first CCSS code of a possibly multi-tagged question.

    Problems carry up to several codes; the first is the one the export lists
    as primary, and using it keeps every question in exactly one content class
    so "same code" is a partition rather than an overlap test.
    """
    codes = question.get("skill_code") or []
    return codes[0] if codes else None


def crossed_sample(questions, quota: dict[str, int], seed: int = 42) -> list:
    """Draw a fixed quota of each format from every fully-crossed CCSS code."""
    cells = defaultdict(list)
    for q in questions:
        if code := primary_code(q):
            cells[(code, q.q_type)].append(q)

    codes = sorted({code for code, _ in cells})
    crossed = [c for c in codes if all(len(cells[(c, f)]) >= quota[f] for f in FORMATS)]
    if not crossed:
        raise SystemExit("No CCSS code carries every format at the requested quota — lower --min_*")

    rng = random.Random(seed)
    sample = []
    for code in crossed:  # sorted, so the draw is reproducible
        for fmt in FORMATS:
            pool = sorted(cells[(code, fmt)], key=lambda q: q["id"])
            sample.extend(rng.sample(pool, quota[fmt]))

    print(f"{len(crossed)} fully-crossed codes of {len(codes)}: {', '.join(crossed)}")
    per_code = sum(quota.values())
    n = len(sample)
    print(f"{n} questions ({per_code}/code = {quota[FILL_IN]} fill-in + {quota[SELECT_ONE]} select-1 "
          f"+ {quota[SELECT_ALL]} select-all) -> {n * n + n:,} scoring pairs per arm")
    return sample


def auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """P(a random positive outranks a random negative), ties counted as half.

    The rank-sum form of the Mann-Whitney statistic, which handles the ties a
    quantized similarity matrix produces without a special case.
    """
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(scores)
    # float(), not the numpy scalar: these end up in json.dump, which has no
    # encoder for np.float32/64 and fails only at write time, after the run.
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def score_arm(pmi_dir: str, questions: list) -> dict:
    """Compute the signal and leak AUCs for one scored arm."""
    n = len(questions)
    pmi = PointwiseMutualInfo.from_shards(pmi_dir, n, n, normalize=False, symmetric=True)
    mat = pmi.pmi_mat

    codes = np.array([primary_code(q) for q in questions])
    fmts = np.array([q.q_type for q in questions])

    # Unordered pairs only: the matrix is symmetrized, so (i, j) and (j, i) are
    # the same observation and counting both would halve the effective sample.
    iu = np.triu_indices(n, k=1)
    same_code = codes[iu[0]] == codes[iu[1]]
    same_fmt = fmts[iu[0]] == fmts[iu[1]]
    scores = mat[iu]

    return {
        "n_pairs": len(scores),
        "signal": auc(scores[~same_fmt], same_code[~same_fmt]),
        "leak": auc(scores[~same_code], same_fmt[~same_code]),
        "n_signal": int((~same_fmt).sum()),
        "n_leak": int((~same_code).sum()),
    }


def report(run_dir: str, data_path: str) -> None:
    """Score every arm under ``run_dir`` and print the comparison table."""
    questions = load_questions(data_path)
    arms = sorted(d for d in glob.glob(os.path.join(run_dir, "*", "pmi"))
                  if glob.glob(os.path.join(d, "predictions_*.pt")))
    if not arms:
        raise SystemExit(f"No scored arms under {run_dir} (expected <run>/<arm>/pmi/predictions_*.pt)")

    rows = []
    width = 12 + 14 + 14
    print(f"\n{len(questions)} questions, {len(arms)} arms")
    print(f"{'arm':<12}{'signal AUC':>14}{'leak AUC':>14}")
    print(f"{'':<12}{'same CCSS':>14}{'same format':>14}")
    print(f"{'':<12}{'| x-format':>14}{'| x-CCSS':>14}")
    print("-" * width)
    for arm_dir in arms:
        arm = os.path.basename(os.path.dirname(arm_dir))
        res = score_arm(arm_dir, questions) | {"arm": arm}
        rows.append(res)
        print(f"{arm:<12}{res['signal']:>14.3f}{res['leak']:>14.3f}")
    print("-" * width)
    print("signal: higher is better (0.5 = no content discrimination)")
    print("leak:   0.5 is the target (0.5 = format carries nothing)")

    out = os.path.join(run_dir, "format-probe.json")
    with open(out, "w") as f:
        json.dump({"data_path": data_path, "n_questions": len(questions), "arms": rows}, f, indent=2)
    print(f"\nwrote {out}")


def main(args):
    if args.score:
        report(args.score, args.data_path)
        return

    questions = load_questions(args.data_path)
    quota = {FILL_IN: args.min_fill_in, SELECT_ONE: args.min_select_one, SELECT_ALL: args.min_select_all}
    sample = crossed_sample(questions, quota, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump_questions(sample, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", required=True, type=str,
                        help="questions.jsonl to draw from (--out) or the probe set that was scored (--score)")
    parser.add_argument("--out", type=str, help="Write the crossed probe set here")
    parser.add_argument("--score", type=str, metavar="RUN_DIR",
                        help="Score every <RUN_DIR>/<arm>/pmi instead of building a set")
    parser.add_argument("--min_fill_in", type=int, default=DEFAULT_QUOTA[FILL_IN])
    parser.add_argument("--min_select_one", type=int, default=DEFAULT_QUOTA[SELECT_ONE])
    parser.add_argument("--min_select_all", type=int, default=DEFAULT_QUOTA[SELECT_ALL])
    parser.add_argument("--seed", type=int, default=42, help="Seed for the within-cell draw")
    parsed = parser.parse_args()
    if not parsed.out and not parsed.score:
        parser.error("pass --out to build the probe set or --score to score a finished run")
    main(parsed)
