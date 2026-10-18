# Examples

A synthetic sample you can run the pipeline on immediately. The 15
questions in `data/sample-mcq.jsonl` are original to this repository
(written for it, derived from no course or dataset) and form three obvious
topic clusters — simple machines, states of matter, food chains — so the
clustering has real structure to find. Each question carries an `lo` field
naming its learning objective (one per topic), as a course question bank
typically does; KCluster passes such extra fields through untouched.

All commands need a local Phi-2 checkout and a GPU
(`pip install "kcluster[local]"`):

    hf download microsoft/phi-2 --local-dir phi-2

## KCluster (EDM 2025): questions → KC models

    export DATA_PATH=examples/data/sample-mcq.jsonl
    export KCLUSTER_RUN_DIR=results/my-run   # one result folder for every step
    kcluster concept --llm_path phi-2 --data_path $DATA_PATH
    kcluster pmi     --llm_path phi-2 --data_path $DATA_PATH
    kcluster build-kc                        # reads and fills the same folder
    kcluster embed   --llm_path phi-2        # optional embedding baselines

All steps share one result folder per dataset, organized by artifact kind:
KC models land in `kc/` (`<ds>_concept-kc.csv` from the concept step,
`<ds>_kcluster-unnorm-kc.csv` from `build-kc`, `<ds>_llm-cosine-kc.csv`
from `embed`; add `--sent_path <sentence-transformer>` to `embed` for the
`sbert-cosine` and `concept-cosine` models too), matrices in `mat/`
(embeddings under `mat/embed/`, the assembled congruity matrix and its raw
score shards under `mat/pmi/`). The concept and congruity steps — usually
separate, long-running jobs — stay paired this way, and `build-kc`/`embed`
need no directory arguments. `--run_dir` does the same per command; without
a run folder the writing steps mint a fresh timestamped folder under
`$KCLUSTER_RESULTS_DIR` (default `results/`). With three planted topics,
the KCluster model should recover roughly three clusters.

No GPU? The same two scoring steps run as Vertex AI batch jobs in your own
GCP project: see `deploy/vertex/README.md`, then `kcluster vertex-launch
--data_path $DATA_PATH` and `kcluster vertex-build-kc`.

## Beyond KC discovery

LO alignment and generate-then-validate (LAK 2026) run on this same sample
through [genval](https://github.com/weiyumou/Genval), which builds on this
package and ships the sample with its standards file.
