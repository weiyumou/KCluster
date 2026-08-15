"""Dataset driver for the 2022 E-learning Design Principles and Methods course.

Course-specific glue for the 2022 offering (DataShop ds5426): parses the
downloaded OLI course HTML with the generic ``oli_html`` loader, keeps only the
questions whose steps ds5426 knows about, and attaches the DataShop step names
(``ds-step-name``) that key those questions to student-step rows.

    python build.py            # reads data/raw/, writes data/processed/ and data/interim/

writes the pair — ``elearning22-mcq.jsonl`` into ``--out_dir`` and the minimal
``elearning22-mcq_student-step-minimal.txt`` into ``--interim_dir`` (the contract
in ``kcluster.io.student_step``) — in one pass, so the questions and the
interaction rows cannot come to describe different step sets. The minimal file is
the tagger's input, one per dataset; each run's tagged file lands in that run's
result dir.

The 2023 offering (ds5843) is a separate driver, ``datasets/elearning23``: the
two share the course HTML and every step of the procedure, and differ only in
how their step names encode the OLI part id and in which KC models they carry.
Both halves of the procedure are therefore in the package —
``oli_html.attach_datashop_steps`` and ``loaders.datashop_export`` — and what
is left here is this offering's own notation.

ds5426's expert models were verified against ``ds5426_kcm.txt``, the KC-model
export of the same steps: the two agree row for row, modulo the ``(unit)``-style
level markers the KC-model export writes into ``Problem Hierarchy`` and the
student-step export does not.
"""

import argparse
import os

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
DS = "elearning22-mcq"

#: The expert KC models ds5426 ships. They define the universe (a step the
#: experts never tagged is not part of the comparison) and ride along in the
#: student-step file for the tagger to count opportunities for.
EXPERT_KC_MODELS = ("LOs-MCQ", "LOs-new-MCQ")

#: Default inputs, under the ``data`` symlink to the dataset directory.
OLI_ROOT = "data/raw/oli-2022-course-export/e_learning_dp-4.2_27gtpdr5/Course_Syllabus"
EXPORT = "data/raw/datashop/ds5426_student_step.txt"


def step_key(raw: str) -> str:
    """The OLI step name a 2022 DataShop step name leads with.

    ``"<question id>_<part id> UpdateRadioButton"`` -> ``"<question id>_<part id>"``,
    which is exactly a question's OLI ``step-name``.
    """
    return raw.split(" ")[0]


def write_elearning22(root_dir: str, export_path: str, out_dir: str, interim_dir: str,
                      kc_models: tuple[str, ...] = EXPERT_KC_MODELS) -> None:
    """Write the 2022 offering's question JSONL and minimal student-step file."""
    export = load_export(export_path, kc_models)
    print(f"** Read {len(export)} student-step rows from {export_path} **")

    questions = attach_datashop_steps(parse_all_mcqs(root_dir), universe_steps(export, kc_models),
                                      raw_key=step_key, step_key=lambda step: step)
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
                        help=f"Path to the ds5426 DataShop student-step export (default: {EXPORT})")
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
    write_elearning22(args.root_dir, args.export, out_dir, interim_dir, tuple(args.kc_models))


if __name__ == "__main__":
    main()
