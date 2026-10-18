# Datasets

KCluster reads a dataset as files that share one stem, the dataset id `<ds>`:

- **`<ds>.jsonl`**: the question bank, in the format the top-level README
  describes. Every pipeline runs on it.
- **`<ds>_student-step-minimal.txt`** (optional): the dataset's interaction log
  reduced to first attempts, carrying no KC model of KCluster's making; the
  contract is in `kcluster.datashop.student_step`. `kcluster tag` joins a run's KC
  models onto it and writes `<ds>_student-step-tagged.txt` into that run's
  result dir — a tagged file belongs to the run whose models it carries — and
  `kcluster fit` scores that.

The stem threads through everything downstream (`elearning22-mcq.jsonl` →
`<result dir>/kc/…/elearning22-mcq_*-kc.csv` →
`<result dir>/elearning22-mcq_student-step-tagged.txt`), so choose it once and
never rename it mid-pipeline.

## Drivers

Whatever turns a raw export into these files is a *driver*. Drivers are not
part of this repository: each encodes the quirks of one export whose raw data
cannot be redistributed, so it lives in a private project beside the data
it builds. The drivers
behind the datasets in KCluster's papers lived in this folder up to 20aaa96 and
remain in the history.

What drivers share ships in the package, under `kcluster.io` and
`kcluster.datashop`:

- format adapters: DataShop exports and KC-model templates (`datashop.export`,
  `datashop.kc_model`) and OLI course HTML (`io.loaders.oli_html`);
- writers and validators: `dump_questions` and `validate_question`
  (`io.jsonl`) for the bank; `save_student_step`, `validate_student_step` and
  `check_coverage` (`datashop.student_step`) for the student-step file.

Write the bank and the student-step file in one pass, so the two cannot come
to describe different question sets: cleaning that drops a question must drop
its interaction rows too.

## Local workspaces

`datasets/<name>/` is room for local work on a dataset, all of it git-ignored:
a `data` link to the dataset's folder, exploratory notebooks (their saved
outputs embed rows of the data), and `jobs/` for the scripts that ran KCluster
on it (they carry cluster paths and cloud project ids). Read through the link,
never write: run outputs belong in the result dir.
