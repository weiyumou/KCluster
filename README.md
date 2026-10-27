# KCluster

[![CI](https://github.com/weiyumou/KCluster/actions/workflows/ci.yml/badge.svg)](https://github.com/weiyumou/KCluster/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15870197-blue.svg)](https://doi.org/10.5281/zenodo.15870197)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LLM-based knowledge component discovery for education.

**KCluster** (EDM 2025) discovers knowledge component (KC) models by
clustering questions with *question congruity*, an LLM-induced similarity
metric: how much more likely one question becomes when another appears
before it. [doi:10.5281/zenodo.15870197](https://doi.org/10.5281/zenodo.15870197)

The package is also the backbone of [genval](https://github.com/weiyumou/Genval),
the Generate-Then-Validate question-generation pipeline (LAK 2026), which
imports its `Question` format, JSONL I/O, PMI core, Phi-2 engine (local GPU
or Vertex AI batch) and versioned [prompt registry](kcluster/core/prompts.py).

## Installation

Not yet on PyPI — install from source:

```bash
git clone https://github.com/weiyumou/KCluster
cd KCluster
pip install -e ".[local]"
```

| extra      | enables |
|------------|---------|
| `local`    | running the pipeline on a local GPU (Phi-2 via transformers + Lightning) |
| `vertex`   | running the scoring steps as Vertex AI batch jobs in your own GCP project — no local GPU needed |
| `gemini`   | the Gemini engine: async batched calls with every raw response kept, and multiple-choice answers parsed from text or token log-probabilities (used by dataset drivers and by genval) |
| `serve`    | `kcluster serve`: the local engine behind an HTTP API, for interactive experiments over an SSH tunnel (see `deploy/vm/`) |

For development: `uv sync --all-extras`, and once per clone
`git config core.hooksPath .githooks` (a pre-commit guard that keeps data
out of the repository).

## Quickstart

Fifteen synthetic sample questions ship in [`examples/`](examples/), with a
[README](examples/README.md) walking through the pipeline. The core
KCluster flow, with a local [Phi-2](https://huggingface.co/microsoft/phi-2)
checkout:

```bash
export DATA_PATH=examples/data/sample-mcq.jsonl
export KCLUSTER_RUN_DIR=results/my-run     # one folder for every step of this run
kcluster concept  --llm_path phi-2 --data_path $DATA_PATH   # concept labels
kcluster pmi      --llm_path phi-2 --data_path $DATA_PATH   # question congruity
kcluster build-kc                                           # clusters congruity into KCs
kcluster embed    --llm_path phi-2                          # embedding baselines (optional)
```

Every step targets one result folder per dataset, organized by artifact
kind — `kc/` for the final KC models, `mat/` for embeddings and congruity
matrices — so steps that run as separate jobs stay together, downstream
commands locate their own inputs, and local and Vertex runs come out
identically shaped.

`build-kc` writes the KCluster model (`<ds>_kcluster-unnorm-kc.csv`)
discovered by affinity propagation over the congruity matrix, alongside the
concept model the concept step produced; `embed` adds cosine baselines from
question and concept embeddings. `build-datashop-kc` fills a
[DataShop](https://pslcdatashop.web.cmu.edu/) KC-model export with a run's
KC models, ready to import for AFM evaluation there, and
`refine-datashop-kc` splits the KCs that fit poorly.

**Another model, another prompt style.** The published prompts are plain
text for a base model to complete, and are what the commands run. Code that
calls the tasks can pass a prompt style instead
([`kcluster.core.prompts`](kcluster/core/prompts.py)): `StockPrompts` with
another remark for a concept name to complete, or `ChatPrompts`, which puts
the same requests in a model's chat template for an instruction-tuned model
and, given the model's end-of-turn token (`end_of_turn=`), ends a generated
name there too.
`extract_concepts`, `PairQuestion` and the Vertex job preparation all take
`prompts=`. The congruity grid is one case of `ScoringGrid`
([`kcluster.tasks.congruity`](kcluster/tasks/congruity.py)), the layout of
(context, text) pairs the scorer and `PointwiseMutualInfo.from_shards`
share; a grid of other texts, such as concept names after each question,
is a subclass that writes its own contexts.

**No GPU?** The scoring steps run as Vertex AI batch jobs in your own GCP
project: deploy the model once ([`deploy/vertex/`](deploy/vertex/README.md)),
then `kcluster vertex-launch` / `vertex-retrieve` / `vertex-build-kc` with
your project settings in a TOML config.

**Question generation** (LAK 2026) lives in
[genval](https://github.com/weiyumou/Genval), which builds on this package.

## Question format

Questions are JSONL, one object per line:

```json
{"id": "q-1", "type": "Multiple Choice",
 "question": {"stem": "…", "choices": [{"label": "a", "text": "…"}]},
 "answerKey": "a"}
```

Extra fields pass through untouched (e.g. `lo` for learning objectives,
`ds-step-name` for DataShop step mapping). Adapters for common sources ship
in the package: `kcluster/datashop/` for DataShop's files, and
`kcluster/io/loaders/` for others (e.g. OLI HTML).

## Repository layout

| path        | contents | shipped in the wheel |
|-------------|----------|----------------------|
| `kcluster/` | the installable package: `core` (Question, PMI, prompts), `engine` (local / Vertex / Gemini), `tasks`, `io`, `datashop` (DataShop's file formats, usable on their own), `output`, CLI commands | yes |
| `examples/` | synthetic sample data + walkthrough | no |
| `datasets/` | the dataset contract: what KCluster reads, and what a driver must produce | no |
| `deploy/`   | the Vertex serving container and deployment scripts | no |

## Data policy

No datasets are distributed in this repository — no student data, DataShop
exports, or course-derived question banks. The sample questions in
`examples/` are synthetic and original to this repo. DataShop datasets
referenced by ID (e.g. ds5426) are available through
[DataShop](https://pslcdatashop.web.cmu.edu/)'s own access process.

## Citation

If you use KCluster in your research, please cite
([CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{wei2025kcluster,
  title     = {{KCluster}: An {LLM}-based Clustering Approach to Knowledge Component Discovery},
  author    = {Wei, Yumou and Carvalho, Paulo and Stamper, John},
  booktitle = {Proceedings of the 18th International Conference on Educational Data Mining (EDM)},
  pages     = {228--240},
  year      = {2025},
  doi       = {10.5281/zenodo.15870197}
}
```

## License

[MIT](LICENSE)
