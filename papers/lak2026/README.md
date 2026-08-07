# Generate-Then-Validate (LAK 2026)

Reproduction scripts for the LAK 2026 paper on generating and validating
multiple-choice questions from course standards. The algorithmic core lives
in the installable `kcluster` package (`tasks/qgen/`, `tasks/classify.py`,
`engine/gemini.py`, `output/qualtrics.py`); this directory holds the
study-specific orchestration on top of it. Per decision D3 this directory is
the source of truth for the thin public paper repo, and the no-data rule
applies to it fully.

## Pipeline

1. **Generate + validate** — `kcluster qgen-generate` / `kcluster qgen-validate`
   (CLI; the study ran Phi-2 with `qs_per_std=100`, batched standards, and
   `prob_thd=0.9`).
2. **Threshold ablation** — `scripts/ablate.py` sweeps `prob_thd` with a
   cheaper single-ordering confidence check.
3. **KC selection** — `scripts/run_concept.py` + `scripts/run_pmi.py` score
   each LO's validated questions, then `scripts/build_kc.py` clusters them
   (normalized PMI), selects MCQs by KC coverage, and writes the
   standards-keyed report.
4. **Expert study** — `scripts/build_qualtrics.py` (the `mix_n_match`
   false-LO manipulation + survey rendering); post-process the exported
   .qsf with `kcluster.output.qualtrics.force_response`.
5. **LLM judges** — `scripts/gemini_logprob_judge.py` (prefilled-turn choice
   log-probs), `scripts/gemini_text_judge.py` (plain-text answers),
   `scripts/gpt_validate.py` (OpenAI Batch API arm), and
   `scripts/gemini_gen_mcq.py` (a structured-output generation baseline).
6. **LO alignment** — `kcluster classify` (CLI).

## Data access

The study's question banks (Podsie courses), standards files, the Qualtrics
raw export, and the partner-school MCQ deliverables are not distributed in
this repository. Contact the authors regarding access.

## Notes

- The original Slurm launchers (cluster-specific paths) and the archived
  one-off variants of the judge scripts remain in the private legacy
  repository; the scripts here are their general forms on the released
  library.
- Analysis notebook and figures are added separately after an output scrub.
