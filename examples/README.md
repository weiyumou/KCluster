# Examples

A synthetic sample you can run the pipelines on immediately. The 15
questions in `data/sample-mcq.jsonl` are original to this repository
(written for it, derived from no course or dataset) and form three obvious
topic clusters — simple machines, states of matter, food chains — so the
clustering has real structure to find. Each question carries an `lo` field
matching `data/standards/actions/sample.txt`.

All commands need a local Phi-2 checkout and a GPU
(`pip install "kcluster[local]"`):

    hf download microsoft/phi-2 --local-dir phi-2

## KCluster (EDM 2025): questions → KC models

    export DATA_PATH=examples/data/sample-mcq.jsonl
    export KCLUSTER_RUN_DIR=results/my-run   # one folder for every step
    kcluster concept --llm_path phi-2 --data_path $DATA_PATH
    kcluster pmi     --llm_path phi-2 --data_path $DATA_PATH
    kcluster build-kc                        # finds concept/ and pmi/ itself

Results are run-major: each step writes to `$KCLUSTER_RUN_DIR/<step>`, so the
concept and congruity steps — usually separate, long-running jobs — stay
paired, and `build-kc` needs no directory arguments. `--run_dir` does the
same per command, and `--output_dir`/`--concept_dir`/`--pmi_dir` still
override individual paths; without a run folder each command mints its own
under `$KCLUSTER_RESULTS_DIR` (default `results/`). `build-kc` writes
`concept-kc.csv`, `question-cosine-kc.csv`, and `pmi-kc.csv`; with three
planted topics, the PMI KC model should recover roughly three clusters.

No GPU? The same two scoring steps run as Vertex AI batch jobs in your own
GCP project: see `deploy/vertex/README.md`, then `kcluster vertex-launch
--data_path $DATA_PATH` and `kcluster vertex-build-kc`.

## LO alignment (LAK 2026): match questions to objectives

    kcluster classify --llm_path phi-2 --data_path $DATA_PATH --lo_type actions

Writes `classified-top3.csv` (top-3 predicted LOs per question) and
`matched-top3.jsonl` (questions whose true `lo` is in the top 3).

## Generate-then-validate (LAK 2026): standards → new MCQs

    kcluster qgen-generate --llm_path phi-2 --std_dir examples/data/standards \
        --config_path examples/data/qgen-config.toml --qs_per_std 8
    kcluster qgen-validate --llm_path phi-2 --root_dir <qgen-run-dir>

`qgen-generate` grows one MCQ per seed incrementally (stem → choices →
answer → explanation) for each standard; `qgen-validate` keeps only
complete questions whose answer clears the permutation-averaged confidence
threshold (`--prob_thd`, default 0.9), sorted by perplexity.

`--config_path` supplies per-step generation settings (TOML or JSON, one
table per step);
[`qgen-config.toml`](data/qgen-config.toml) holds the LAK 2026 study's.
It is effectively required: asking for more than one question per standard
needs sampling on the stem step, which the transformers defaults (greedy)
cannot do.
