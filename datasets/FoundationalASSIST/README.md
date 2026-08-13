# Foundational ASSIST (ASSISTments)

Driver for the Foundational ASSIST release: middle-school mathematics problems
(fill-in-the-blank, select-1, select-all) with CCSS skill tags and a student
interaction log. The dataset is **gated**, licensed CC-BY-NC-4.0, and carries a
data-security undertaking — keep every raw and processed file outside this
repository and never commit any of it.

`data` is a symlink to the dataset directory, which holds `raw/` (the release's
own `Code/`, `Data/`, and `README.md`), `interim/`, and `processed/`. The
release's HTML-cleaning code is vendored here as `clean_utils.py` — adapted
from `raw/Code/clean_utils.py`, with provenance, license (CC-BY-NC-4.0, unlike
the MIT package) and the changes made documented in the module docstring — so
the driver no longer imports code out of the gated data directory.

## Pipeline

    python processing.py                  # -> interim/{problems.csv, interactions.csv}
                                          #    processed/{foundational-assist.jsonl,
                                          #               foundational-assist_student-step.txt}
    python answerability.py --dry_run     # inspect the prompts, no API calls

`processing.py` writes all four artifacts in one
pass — generating the questions or the student-step file separately would let
them drift out of step with `problems.csv`. The jsonl stem `foundational-assist`
is the dataset id: every KC artifact downstream inherits it as a filename
prefix. Problems are typed by `Problem Type` so both multiple-choice families
keep the `Multiple Choice …` prefix that `validate_question` guards on.

## Student-step file

`foundational-assist_student-step.txt` follows the minimal student-step
contract (`kcluster.io.student_step`): one row per student × problem holding
the **first attempt** — the interaction with the earliest `end_time`, ties
broken by log id (2.6% of student–problem pairs have more than one log row;
the export does not say whether repeats are new encounters or retries, so
everything after the first attempt is dropped, not guessed at). Outcomes keep
the platform's scoring: `discrete_score` 1 → `correct`, 0 → `incorrect`.
ASSISTments logs hint use and answer viewing separately from the score;
DataShop would code a hint-first attempt as `hint` (a failure in AFM), but
88.5k of 88.8k hint-assisted rows and 375.2k of 375.5k saw-answer rows are
scored 0 anyway, so the mapping moves ~0.03% of outcomes and we keep the
source's semantics. `Step Name` holds the question id (`fa-<problem_id>`),
`Problem Name` copies it, and the expert CCSS model rides along as
`KC (CCSS)` — with **no** `Opportunity` column, because the KC tagger owns
opportunity counting for expert and generated models alike.

The cleaning rules were derived in an exploratory notebook that is **not** in the
repository: its saved outputs embed problem text and interaction rows from a
gated dataset, so `datasets/**/*.ipynb` is git-ignored. `processing.py` is
therefore the record of that work — each numbered step names the defect it
corrects, and the counts it pins (`EXPECTED_TRUNCATED_KEYS`, the assertion in
step 11) are the evidence that the defect is still the size it was.

Three drops are optional, so the same driver produces every variant:

| invocation | problems | composition |
|---|--:|---|
| `--keep_image_problems --keep_unanswerable` | 3,298 | 2,141 fill-in · 784 select-1 · 373 select-all |
| `--keep_unanswerable` | 2,563 | 1,758 fill-in · 572 select-1 · 233 select-all |
| *(default)* | **2,019** | 1,303 fill-in · 489 select-1 · 227 select-all |
| `--drop_select_all` | **1,792** | 1,303 fill-in · 489 select-1 |

The default drops the 544 problems in `unanswerable.py` — the output of the
answerability screen and its manual review. `--drop_select_all` additionally
removes select-all-that-apply, whose multi-label keys need set-valued scoring
the rest of the pipeline does not model.

## Answerability screen

Some problems cannot be answered from their own text: they need a figure that
is not described, or a quantity introduced in an earlier part the export does
not ship. `answerability.py` asks Gemini to answer every problem and records
two independent signals — the answer itself (exact match against the key, with
choice probabilities for select-1 via the prefilled-turn trick) and a direct
Yes/No self-containment probe. Requires the `gemini` extra and `GOOGLE_API_KEY`
(or `--vertexai` with your own project).

    python answerability.py --limit_per_type 20    # smoke test first
    python answerability.py                        # full screen, ~2.5k problems

**Log-probabilities need Vertex AI and a Gemini 2.5 model.** The Developer API answers
`Logprobs is not enabled` for every model, and Gemini 3.x rejects the option on both
endpoints, so a newer model can only be screened in text mode — answers still get their
exact-match verdict, but there is no choice distribution and self-containment collapses to
a hard Yes/No. A preflight call settles this before the run spends anything and falls back
automatically; `--require_logprobs` turns the fallback into an error instead. For the
probability signal:

    python answerability.py --vertexai --project <gcp-project> --model gemini-2.5-flash

The report's `mode` column records which path produced each row.

Output lands in `interim/answerability/<timestamp>/`: the raw responses for
each arm plus `answerability.csv`, one row per problem with the model's answer,
`exact_match`, the probabilities, and a `flag_reason`. A run can be continued
with `--resume_from <earlier dir>`, which reuses every non-error response.

Treat the flags as a review queue, not a delete list: a failure conflates
missing context with a genuinely hard problem, a tolerance-graded key (`9pi`,
`40000` for an estimation item), and an actual key error. Matching is
deliberately strict — only the `" || "` (alternative answers) and `", "`
(multi-blank) delimiters are interpreted, never numeric tolerance or
fraction/decimal equivalence — so `model_answer_raw` is kept for re-scoring
under looser rules without re-running the screen.

## Reviewing what the screen could not settle

`review_app.py` serves the flagged problems one at a time on `localhost`, with
the problem text, the stored key, Gemini's answer, and what students who were
marked correct actually typed side by side. Verdicts (keep / drop / fix the key
/ unsure, plus a corrected key and a note) are written to disk on every click,
so a pass can be interrupted and resumed.

    python review_app.py --triage <run>/triage.csv

It binds to `127.0.0.1` and makes no external requests — the dataset's terms
require the problem text stay on secured systems, so it must never be served on
a public interface or published to a hosted page. Results land beside the
triage file as `review-decisions.json` and `review-decisions.csv`.

## Known data properties

- 9 select-1 problems keep a `____` blank in the stem (converted drop-downs);
  harmless, since the options are rendered explicitly.
- 738 problems (22.2%) depend on an image and are dropped by default; pass
  `--keep_image_problems` to retain them.
- Student `answer_text` in the interaction log often disagrees with the key by
  rendering only (`3^2` vs `3²`, `(5/16)` vs `5/16`) or stores an option label
  rather than the option text. Normalize before matching interactions to keys.
- The keys themselves are inconsistent about thousands separators — 20 fill-in
  keys write `1,000` while 49 write `17160` — so no answering prompt can match
  both conventions and strict scoring will flag some correct answers. Re-score
  from `model_answer_raw` rather than loosening the screen.
- Seven fill-in keys were stored as the platform's 9-decimal-place rounding of a
  repeating decimal (`-0.055555556` for `-1/18`), unmatchable by anyone writing
  the exact fraction. Step 11 recovers the fraction and keeps the decimal as a
  `||` alternative; `EXPECTED_TRUNCATED_KEYS` pins the count for this export.
- ASSISTments graded with algebraic equivalence, not string match, so
  `discrete_score == 1` does **not** imply the student's text equals the key.
  A sweep of all 2,525 problems with correct interactions found the modal
  student answer equal to the key for 1,919, equivalent to it for a further 194,
  and genuinely different for 73 — mostly the interaction log storing an option
  label, truncated option text, or tolerance-graded estimation items.
