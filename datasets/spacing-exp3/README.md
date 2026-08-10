# Spacing study, experiment 3

Driver for the ds6824 transaction export (Podsie). `processing.py` cleans the raw
transaction file and extracts the study's multiple-choice bank as question JSONL.

Run from this directory:

    python processing.py --raw_tx_path raw_data/ds6824_tx.txt --output_dir data/

Produces two files:

- `data/mcq.jsonl` -- **222 questions**, four choices each.
- `data/mcq-tx.txt` -- the matching cleaned transactions, **151,362 of the export's
  154,237 rows**, from 323 students across five phases (pretest1, pretest2, learning,
  posttest1, posttest2), ~681 responses per item.

The two join on `Step Name` / `CF (Question ID)`, both of which carry the question ID
(`mcq-N`): the export's own `Step Name` is the constant "Submit Answer" and identifies
nothing, so it is overwritten with the ID exactly as exp2 does, which is what a DataShop
KC model import expects. Dropped from the transaction copy are the filler stems and the
superseded exemplar variants described below; nothing else. The file keeps DataShop's
tab-delimited layout and its duplicated `Condition Name` / `Condition Type` header pairs
(spacing condition first, algorithm condition second).

Timed-out attempts are **kept** -- 17,963 rows with a blank `Input` (no answer given) and
2,097 more that ran out the clock but were still answered -- because a non-response is
part of the response record; filter on `CF (Timed Out)` if a model should not see them.
Two rows in the whole file disagree with themselves (`Outcome` is INCORRECT while `Input`
equals the answer key); every other row is consistent, and every `Input` in the file is
one of its question's four declared options.

## Why this is a separate driver from spacing-exp2

The two exports share a study design but not a template, so `spacing-exp2/processing.py`
does not transfer. ds6824 has **no** `CF (Answer Options)`, `CF (Correct Answer Options)`,
`CF (Stage)`, `Level (Course)`, `CF (Topic Text)`, `CF (Standard Name)`, or
`KC (question_group)`. What it does have:

- **Options are reconstructed from responses.** Each stem's choice set is the set of
  `Input` values students actually selected, so an option nobody ever picked would be
  invisible. In practice this recovers exactly four choices for all 222 items, and the
  key (`CF (Exemplar Answer)`) is always among them.
- **`CF (Question Id)` is unusable.** One id spans up to five unrelated stems, so items
  are keyed on problem text and renumbered `mcq-N` in order of first appearance, as in
  exp2. Note that `Problem Name` and `CF (Full Problem Name)` are truncated at different
  widths (240 vs 255 characters) and disagree on 1.4% of rows; the stem comes from the
  longer one.
- **`KC (Unique-step)` is not a KC model.** It is phase x problem (1,000 values over 304
  stems), not a knowledge component.

Two filters do the cleaning work. Stems answered by fewer than `--min_students` (default
20) students are one-off filler from unrelated assignments -- 82 stems, 96 rows, covering
history, vocabulary, and geometry, and including every image-bearing stem in the export.
Twelve remaining stems carry more than one `CF (Exemplar Answer)`: a few are single-row
export glitches where the exemplar belongs to a different question, and one
("What is an invasive species?") is a genuine rewrite with a fresh option set.
`--variant_policy dominant` (default) keeps the variant with the most responses;
`latest` keeps the most recently answered one. Options are read only from rows sharing
the kept exemplar, so option sets never mix across versions. One stem (`mcq-39`) is cut
off at the 255-character DataShop export limit.

## Scope: multiple choice only

The export also holds 1,511 short-answer items, but each was answered **exactly once** --
they are generated per student per posttest, with an LLM exemplar answer. They are a
question bank without a response distribution, so they are excluded here; add a second
output if a text-only clustering run needs them.

## What this workspace cannot do

There is **no reference KC labelling** in ds6824 -- no standard, topic, or question group.
`spacing-exp2/create_ds_kc.py` therefore has no counterpart here: only the DataShop default
models (Single-KC, Unique-step) can be built, and neither is a usable comparison KC model.
Evaluating a KCluster model on this workspace has to run through the response data
(learning curves, AFM fits) rather than agreement with a human labelling.

Compared with exp2's 7,069 questions over 13 courses, this is a small bank with dense
per-question response data rather than a large one. It is also a single subject
(middle-school science) with only one question format in the extracted set, so the format
leakage that motivates `--residualize` on exp2 (decision D9) does not arise here.
