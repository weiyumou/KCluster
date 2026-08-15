"""Dataset driver for the 2023 E-learning Design Principles and Methods course.

The 2023 offering (DataShop ds5843) of the course ``datasets/elearning22``
covers: same OLI content, re-run with a new cohort, so the dataset directory
symlinks the 2022 course export rather than holding a second copy. This driver
is therefore the 2022 one minus everything the two share — the procedure lives
in ``oli_html.attach_datashop_steps`` and ``loaders.datashop_export`` — leaving
two differences:

- **Step names encode the OLI part id differently.** 2022 leads with it
  (``"<question>_<part> UpdateRadioButton"``); 2023 buries it after a ``part``
  marker (``"Activity <name>, part <part> Multiple choice submission"``), and
  the question's own step name has to be reduced to match.
- **Different expert KC models.** A KC model kept in the raw student-step
  export is an expert model by convention: it defines the step universe and
  rides along for the tagger, whatever route it took into DataShop. For ds5426
  that is the course's LO taggings; for ds5843 it is the two EDM 2025 models on
  the MCQ universe named in :data:`EXPERT_KC_MODELS`, which reproduce the
  602-step universe every ``-MCQ`` model in the export tags.

  The export's ``KC (Single-KC)`` and ``KC (Unique-step)`` columns are
  deliberately *not* carried: the tagger builds baselines under those names for
  every file it tags, and would refuse a file that already claims them.

    python build.py            # reads data/raw/, writes data/processed/ and data/interim/

writes ``elearning23-mcq.jsonl`` into ``--out_dir`` and the minimal
``elearning23-mcq_student-step-minimal.txt`` into ``--interim_dir`` in one pass,
so the questions and the interaction rows cannot come to describe different step
sets.
"""

import argparse
import os
import re

from kcluster.io.jsonl import dump_questions
from kcluster.io.loaders.datashop_export import load_export, reduce_to_steps, universe_steps
from kcluster.io.loaders.oli_html import attach_datashop_steps, parse_all_mcqs
from kcluster.io.student_step import (
    MINIMAL_SUFFIX,
    check_coverage,
    save_student_step,
    validate_student_step,
)

#: The jsonl stem is the dataset id: every KC artifact downstream inherits it.
DS = "elearning23-mcq"

#: The expert KC models ds5843 carries — kept in the raw export, so they play
#: the expert role (see the module docstring): they define the universe and
#: ride along in the student-step file for the tagger to count opportunities for.
EXPERT_KC_MODELS = ("v1-prompt-CTAmultimedia-MCQ", "v2-combined-MCQ")

#: Default inputs. The OLI export is the 2022 course directory, symlinked.
OLI_ROOT = "data/raw/oli-2022-course-export/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
EXPORT = "data/raw/datashop/ds5843_student_step.txt"

PART_MARKER = re.compile(r"(?<= part )\S+")


def raw_key(raw: str) -> str | None:
    """The OLI part id a 2023 DataShop step name carries after its ``part`` marker.

    ``None`` for a step name without one, which is then not matched to any
    question rather than crashing the join.
    """
    return match.group(0) if (match := PART_MARKER.search(raw)) else None


def step_key(step: str) -> str:
    """The part id in a question's OLI step name, ``"<question id>_<part id>"``."""
    return step.split("_")[-1]


def write_elearning23(root_dir: str, export_path: str, out_dir: str, interim_dir: str,
                      kc_models: tuple[str, ...] = EXPERT_KC_MODELS) -> None:
    """Write the 2023 offering's question JSONL and minimal student-step file."""
    export = load_export(export_path, kc_models)
    print(f"** Read {len(export)} student-step rows from {export_path} **")

    questions = attach_datashop_steps(parse_all_mcqs(root_dir), universe_steps(export, kc_models),
                                      raw_key=raw_key, step_key=step_key)
    questions_path = os.path.join(out_dir, f"{DS}.jsonl")
    dump_questions(questions, questions_path)
    print(f"** Saved {len(questions)} questions to {questions_path} **")

    ss = reduce_to_steps(export, {step for q in questions for step in q["ds-step-name"]})
    validate_student_step(ss)
    uncovered = check_coverage(questions, ss)
    assert not uncovered, f"{len(uncovered)} question(s) have no student-step rows: {uncovered[:5]}"
    ss_path = os.path.join(interim_dir, f"{DS}{MINIMAL_SUFFIX}")
    save_student_step(ss, ss_path)
    print(f"** Saved {len(ss)} student-step rows, {len(export) - len(ss)} dropped as unresolvable "
          f"({ss['Anon Student Id'].nunique()} students, "
          f"{ss['First Attempt'].eq('correct').mean():.1%} correct) to {ss_path} **")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_dir", default=OLI_ROOT, type=str,
                        help=f"Root of the downloaded OLI course HTML (default: {OLI_ROOT})")
    parser.add_argument("--export", default=EXPORT, type=str,
                        help=f"Path to the ds5843 DataShop student-step export (default: {EXPORT})")
    parser.add_argument("--out_dir", default="data/processed", type=str,
                        help="Where to write the question JSONL (default: data/processed)")
    parser.add_argument("--interim_dir", default="data/interim", type=str,
                        help="Where to write the minimal student-step file (default: data/interim)")
    parser.add_argument("--kc_models", nargs="+", default=list(EXPERT_KC_MODELS), type=str,
                        help="Expert KC models defining the universe, carried into the student-step file")
    args = parser.parse_args()

    out_dir, interim_dir = os.path.abspath(args.out_dir), os.path.abspath(args.interim_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)
    write_elearning23(args.root_dir, args.export, out_dir, interim_dir, tuple(args.kc_models))


if __name__ == "__main__":
    main()
