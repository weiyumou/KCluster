# KCluster

[![CI](https://github.com/weiyumou/KCluster/actions/workflows/ci.yml/badge.svg)](https://github.com/weiyumou/KCluster/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2505.06469-b31b1b.svg)](https://arxiv.org/abs/2505.06469)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LLM-based knowledge component discovery and question generation for education.

- **KCluster** (EDM 2025) discovers knowledge component (KC) models by
  clustering questions with *question congruity*, an LLM-induced similarity
  metric: how much more likely one question becomes when another appears
  before it. [arXiv:2505.06469](https://arxiv.org/abs/2505.06469)
- **Generate-Then-Validate** (LAK 2026) generates multiple-choice questions
  from learning objectives, using a small language model as both an
  expansive generator and a selective validator.

Both run on the same backbone: one `Question` format, one Phi-2 engine
(local GPU or Vertex AI batch), one PMI core, and a versioned
[prompt registry](kcluster/core/prompts.py).

## Installation

Not yet on PyPI — install from source:

```bash
git clone https://github.com/weiyumou/KCluster
cd KCluster
pip install -e ".[local]"
```

| extra      | enables |
|------------|---------|
| `local`    | running the pipelines on a local GPU (Phi-2 via transformers + Lightning) |
| `vertex`   | running the scoring steps as Vertex AI batch jobs in your own GCP project — no local GPU needed |
| `gemini`   | the async Gemini engine used by the LAK 2026 judge and baseline scripts |
| `datashop` | parsing LearnSphere/DataShop cross-validation results |

For development: `uv sync --all-extras`, and once per clone
`git config core.hooksPath .githooks` (a pre-commit guard that keeps data
out of the repository).

## Quickstart

Fifteen synthetic sample questions ship in [`examples/`](examples/), with a
[README](examples/README.md) walking through all three pipelines. The core
KCluster flow, with a local [Phi-2](https://huggingface.co/microsoft/phi-2)
checkout:

```bash
export DATA_PATH=examples/data/sample-mcq.jsonl
export KCLUSTER_RUN_DIR=results/my-run     # one folder for every step of this run
kcluster concept  --llm_path phi-2 --data_path $DATA_PATH   # concept labels + embeddings
kcluster pmi      --llm_path phi-2 --data_path $DATA_PATH   # question congruity
kcluster build-kc                                           # finds concept/ and pmi/ itself
```

Results are run-major — `<run>/{concept,pmi,kc}` — so steps that run as
separate jobs stay together and downstream commands locate their own inputs.

`build-kc` writes labeled KC models (`pmi-kc.csv`, `concept-kc.csv`, an
embedding baseline) discovered by affinity propagation over the congruity
matrix. `build-datashop-kc` and `refine-datashop-kc` then insert and refine
KC models in [DataShop](https://pslcdatashop.web.cmu.edu/) format for AFM
evaluation.

**No GPU?** The scoring steps run as Vertex AI batch jobs in your own GCP
project: deploy the model once ([`deploy/vertex/`](deploy/vertex/README.md)),
then `kcluster vertex-launch` / `vertex-retrieve` / `vertex-build-kc` with
your project settings in a TOML config.

**Question generation** (LAK 2026): `kcluster qgen-generate` grows MCQs
incrementally from course standards, `kcluster qgen-validate` keeps only
questions whose answer clears a permutation-averaged confidence threshold,
and `kcluster classify` aligns questions with learning objectives via a
rectangular LO-by-question congruity grid.

## Question format

Questions are JSONL, one object per line:

```json
{"id": "q-1", "type": "Multiple Choice",
 "question": {"stem": "…", "choices": [{"label": "a", "text": "…"}]},
 "answerKey": "a"}
```

Extra fields pass through untouched (e.g. `lo` for learning objectives,
`ds-step-name` for DataShop step mapping). Format adapters for public
datasets live in `kcluster/io/loaders/` (e.g. ScienceQA, OLI HTML).

## Repository layout

| path        | contents | shipped in the wheel |
|-------------|----------|----------------------|
| `kcluster/` | the installable package: `core` (Question, PMI, prompts), `engine` (local / Vertex / Gemini), `tasks`, `io`, `output`, CLI commands | yes |
| `examples/` | synthetic sample data + walkthrough | no |
| `papers/`   | paper-specific reproduction scripts (`lak2026/`) | no |
| `datasets/` | per-dataset ETL workspaces (spacing, webwork, elearning, …) | no |
| `deploy/`   | the Vertex serving container and deployment scripts | no |

## Data policy

No datasets are distributed in this repository — no student data, DataShop
exports, or course-derived question banks. The sample questions in
`examples/` are synthetic and original to this repo. DataShop datasets
referenced by ID (e.g. ds5426) are available through
[DataShop](https://pslcdatashop.web.cmu.edu/)'s own access process; for the
LAK 2026 study artifacts, see [`papers/lak2026/`](papers/lak2026/README.md).

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
