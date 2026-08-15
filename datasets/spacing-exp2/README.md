# Spacing study

Driver for the spacing-study transaction exports (Podsie; DataShop ds6527-era
templates). `processing.py` cleans a raw transaction file (column selection,
stage filter, stable question IDs derived from course name + problem text +
answer options) and writes the D12 driver artifacts in one pass: one
`<Course>.jsonl` per course into `processed/`, because that is the unit KCluster
runs on, and into `interim/` the whole study's untagged
`spacing-exp2_student-step-minimal.txt` beside the cleaned transaction file it
was reduced from. `create_ds_kc.py` populates the DataShop KC models (defaults,
Standard, Q-Group, plus the KCluster/Concept results).

Run from this directory, e.g.:

    python processing.py --raw_tx_path data/raw/<export>.txt --output_dir data/processed

The expert KC models the export ships (topic text and question group) are
collapsed to one label per step, majority wins, because AFM and DataShop both
read a KC model as a map from steps to KCs and every generated model here is one
label per question. Nine disagreements are affected and each is logged. Eight
are defects — two stray transactions, and `QG-1253`, a duplicate of `QG-797`
over the identical four steps, which the collapse retires. The ninth is real:
`tof-220` is the true-false member of two question-group quartets, so it keeps
both labels as `QG-954~~QG-955` rather than being taken from one of them.

Tagging the interim file with the KC models turns it into the DataShop-style
file AFM/PFA packages consume:

    kcluster tag --ss_path data/interim/spacing-exp2_student-step-minimal.txt \
        --run_dir <results>/spacing-exp2-vertex

which writes `spacing-exp2_student-step-tagged.txt` into that work dir — a
tagged file belongs to the run whose KC models it carries.

The KC models are per course, so tagging the one undivided student-step file
takes them concatenated. `--run_dir` on a work dir picks up each course result
dir's own `kc/*-kc.csv` and concatenates the 13 parts of each model in memory;
no dataset-level KC table is written, since one on disk would be a copy to keep
in sync with the parts it came from.

Courses keep **disjoint KC spaces**: each part's labels are prefixed with its
course (`EPLA-Physics: Newton's third law`), because clustering ran per course
and 366 of the 3,314 generated concept labels are spelled the same in more than
one — three courses write some form of Newton's third law. Without the prefix
those steps would share one KC and one set of AFM parameters despite never
having been compared. The expert models need no such treatment: the export's
topic texts and question groups are already course-specific (0 of 312 topics and
0 of 1,776 question groups occur in two courses).

Spaces in a course name become hyphens, matching the stem the rest of the
pipeline derives (`vertex-build-kc`'s result dir, and the `<ds>_<model>-kc.csv`
KC files read back out of it). The question ids are unique across
all 13 courses, which is what lets `ds-step-name` alone key a student-step row
to its question: no course or hierarchy field takes part in the join, and the
interaction log therefore needs no splitting. Tagging it takes the courses' KC
models concatenated, one frame per model. A student-step row is one
(course, step, student, **session**) encounter, carrying the export's two expert
KC models, topic text and question group.

### What counts as one encounter

DataShop's rollup gives a step one row per *encounter* and records only how that
encounter opened (`First Attempt`); the attempts a student makes after seeing
feedback belong to the encounter they followed, and their evidence of learning
is the next encounter's first attempt. DataShop takes the boundary from logged
events rather than from a clock — the only threshold in its docs is the 10
minutes past which a *transaction duration* is reported as null — and this export
logs one too, `Session Id`, so the driver keys on that.

It matters here because both kinds of repeat are common on the same step in the
same stage. Of the export's 77,629 repeat transactions, the 52% under ten minutes
apart never cross a session (a retry after feedback), the 45% more than a day
apart cross one 99.8% of the time (the study's spaced re-presentation), and only
856 fall in between — precisely the cases a time threshold would have to guess
at, and the ones the session id settles outright.

Keying on the stage instead, as this driver first did, silently merged every
spaced re-encounter inside the learning stage: 190,821 rows rather than 226,622,
with the learning stage's repeats replaced by their first sitting alone. The
recovered rows carry the effect the study exists to measure — first-attempt
accuracy across spaced encounters runs 57.1% → 58.8% → 63.9% → 67.4% → 69.6%.
Stage is not part of the key at all now: no session spans two stages for one
student and question, so it is implied by the session and kept only as a label
of the design.

This pipeline supersedes the older `kcluster/datasets/spacing.py` extractor
from the legacy main repo (pre-cleaning export schema, `Input`-grouped
choices), which was not ported.

## Format balance, and why it matters here

The 13 courses hold 7,069 questions, balanced almost exactly four ways by
question format (1,767 multiple choice / 1,766 short answer / 1,768 true-false /
1,768 fill-in). That balance is deliberate in the study design, and it makes thisthe worst case for format leakage in question congruity: pairs score congruent
partly for sharing a format, and affinity propagation then recovers format
families rather than KCs. Cluster with `--residualize` (decision D9) and compare
against the uncorrected model rather than assuming either is right.

Job scripts that ran KCluster on this workspace live in the git-ignored `jobs/`
(see `datasets/README.md`); `jobs/vertex-kcluster.sh` drives the Vertex AI path,
one batch job per course.
