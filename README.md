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
| `gemini`   | the Gemini engine: async batched calls with every raw response kept, and multiple-choice answers parsed from text or token log-probabilities (used by dataset drivers and by genval); it takes the same prompts as the local engine, strings or conversations |
| `serve`    | `kcluster serve`: the local engine behind an HTTP API, for interactive experiments over an SSH tunnel (see `deploy/vm/`) |
| `multimodal` | showing a bank's figures to a model that takes images (Gemma 4): decoding the files, and the resizing the image processor does through torchvision |

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

**Figures.** With a model that takes images (Gemma 4, `pip install -e
".[local,multimodal]"`), the same three steps show each question's figures
to the model: `--llm_path gemma-4-12B`, with a small `--batch_size`;
`--figure_dir` names the folder the bank's figure paths are relative to when
it is not the bank's own, and `--ignore_figures` runs the text-only baseline.
See *Question format*.

**Another model, another prompt style.** The published prompts are plain
text for a base model to complete, and are what the commands run. Code that
calls the tasks can pass a prompt style instead
([`kcluster.core.prompts`](kcluster/core/prompts.py)): `StockPrompts` with
another remark for a concept name to complete, or `ChatPrompts`, which puts
the same requests in a conversation for an instruction-tuned model. A style
also reads the name out of the completion: the chat style takes out the
bold and quotation marks such a model puts around a name. The
engine takes a prompt either way: a string, read as it is, or a conversation
as `{"role", "content"}` messages, which it writes in the model's own chat
template with the model's turn opened (a last assistant message is the
prefill the model continues). A served model renders messages with its own
template, so a client needs no tokenizer. `extract_concepts`, `PairQuestion`
and the Vertex job preparation all take `prompts=`; the steps under them,
`complete_names` and `embed_texts`, take any prompts at all. A context and
the text scored after it are tokenized together, as the model reads them,
and the text's tokens found by their character offsets; the commands'
`--add_bos` puts the model's BOS token before every string prompt when the
tokenizer adds none (Gemma 4's instruction-tuned tokenizer; its base one adds
its own). Images are the engine's too: a prompt or a text places one with
the model's image token (`llm.image_token`), an image part in a message is
the same place written the template's way, and the images go beside the
prompt (`images=`, `context_images=`), paths, bytes or PIL images, which the
HTTP client sends as data URIs. An image's tokens are context: never scored,
never pooled. The tasks put the token at each `[figure]` marker when given a
figure root (`figure_dir=`) and show no figure otherwise.

**Precision.** A command loads a model in the precision its config names
(Phi-2 fp16, as the published runs scored it; Gemma bf16, since it overflows
in fp16) unless `--dtype` says otherwise. The output head runs in float32
whatever the model's precision: the last hidden states are projected onto
the vocabulary in float32, a chunk at a time, and the loss is summed there,
so a score is exact to float32 given the model's hidden states. Without
that, a bf16 logit near 25 is good to 0.125 nats and rounds differently with
the batch shape, and the scores moved by that much with the batch they were
in. A padded row is scored at its own positions, counted from its first real
token, as a rotary model reads shifted positions as the same text in exact
arithmetic but not in bf16. What remains is the bf16 forward's own variation
with the batch shape, a few tenths of a nat per text for Gemma 4 on a GPU,
which only a float32 forward would remove; a run repeats exactly when every
batch has the same shape (`--batch_size`, `--pad_to_multiple_of`).

**Any scoring grid.** The congruity grid is one case of `ScoringGrid`
([`kcluster.core.grid`](kcluster/core/grid.py)), the layout of (context,
text) pairs the scorer, the Vertex jobs and `PointwiseMutualInfo.from_shards`
share: `m` texts in the columns after each of `n` contexts in the rows, plus
each text after its marginal context. A grid of other texts, such as concept
names after each question, is built from two functions of the cell, or by a
subclass that writes its own contexts; the rectangular PMI is the raw
estimator. `kcluster pmi` counts the grid's longest pair before scoring and
stops if it is over the model's context length, since the engine neither
truncates nor warns.

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

A question that shows figures writes `[figure]` where each one appears
(in the stem or a choice) and lists their files in `figures`, in that
order, as paths relative to the bank's folder (or to `--figure_dir`):

```json
{"id": "q-2", "type": "Short Answer",
 "question": {"stem": "[figure]\nThe graph of f is shown. f'(2) = ___"},
 "answerKey": "3", "figures": ["figures/q-2-1.png"]}
```

A text-only model reads the marker as those eight characters; a model that
takes images (Gemma 4, the `multimodal` extra) sees the image in its place.
The scored text is the same either way: figures are context, never part of
what is scored.

## Repository layout

| path        | contents | shipped in the wheel |
|-------------|----------|----------------------|
| `kcluster/` | the installable package: `core` (Question, PMI, prompts, the scoring grid), `engine` (local, with images / Vertex / Gemini, and the HTTP server and client), `tasks`, `io`, `datashop` (DataShop's file formats, usable on their own), `output`, CLI commands | yes |
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
