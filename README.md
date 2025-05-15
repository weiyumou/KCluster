<div align="center"> 

# KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery

### Yumou Wei, Paulo Carvalho, John Stamper

### Carnegie Mellon University, Pittsburgh, PA, USA

### To appear in the proceedings of the 18th *International Conference on Educational Data Mining* (EDM 2025)


[![Paper](http://img.shields.io/badge/arXiv-2505.06469-B31B1B.svg)](https://arxiv.org/abs/2505.06469)
[![Conference](http://img.shields.io/badge/EDM-2025-4b44ce.svg)](https://educationaldatamining.org/edm2025/)

</div>


## Abstract
Educators evaluate student knowledge using knowledge component (KC) models that map assessment questions to KCs. Still, designing KC models for large question banks remains an insurmountable challenge for instructors who need to analyze each question by hand. The growing use of Generative AI in education is expected only to aggravate this chronic deficiency of expert-designed KC models, as course engineers designing KCs struggle to keep up with the pace at which questions are generated. In this work, we propose KCluster, a novel KC discovery algorithm based on identifying clusters of congruent questions according to a new similarity metric induced by a large language model (LLM). We demonstrate in three datasets that an LLM can create an effective metric of question similarity, which a clustering algorithm can use to create KC models from questions with minimal human effort. Combining the strengths of LLM and clustering, KCluster generates descriptive KC labels and discovers KC models that predict student performance better than the best expert-designed models available. In anticipation of future work, we illustrate how KCluster can reveal insights into difficult KCs and suggest improvements to instruction.

## Setup
This repository contains the code and data for `KCluster`. To set up `KCluster` for your workflow, please follow these steps.

### 1. Clone the repository
```bash
git clone https://github.com/weiyumou/KCluster.git
cd KCluster
```

### 2. Create a virtual environment
We recommend using `conda` or `venv` to create a virtual environment.

If you are using `conda`, you can create a virtual environment with the following command:
```bash
conda create -n kcluster python=3.12
conda activate kcluster
```

If you are using `venv`, you can create a virtual environment with the following command:
```bash
python -m venv kcluster
source kcluster/bin/activate
```

`KCluster` requires Python 3.12.

### 3. Install dependencies
You may try to install all the dependencies using `pip`:
```bash
pip install -r requirements.txt
```
But if it fails, you can install the dependencies one by one. In general, `KCluster` requires:
 - `torch` and `lightning` for custom PyTorch implementations;
 - `transformers` and `accelerate` for HuggingFace LLMs;
 - `pandas` (and `numpy`) for data manipulation;
 - `scikit-learn` for clustering algorithms;
 - `sentence-transformers` for concept embeddings. 

We also used `jupyterlab` to develop an early prototype. If you want to reuse our code for parsing [DataShop](https://pslcdatashop.web.cmu.edu/) HTML files, you may need to install `beautifulsoup4`.


### 4. Download the LLM
We used `Phi-2` developed by [Microsoft](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) and made available through [HuggingFace](https://huggingface.co/microsoft/Phi-2). According to the model page, please clone the repository to download `Phi-2` using the following commands (make sure you have `git-lfs` installed):
```bash
git lfs install
git clone https://huggingface.co/microsoft/phi-2
```
See [here](https://huggingface.co/microsoft/phi-2/tree/main?clone=true) for more details. We use the environment variable `$LLM_PATH` to denote the path to the model directory (e.g., `/home/user/phi-2`).

## Workflow
### 1. Prepare your data
`KCluster` takes a list of questions as input and returns the KCs as output. The input should be a `.jsonl` file where each line is a JSON object representing a question. The JSON object should be structured as follows:
```python
{
  "id": str,  # Question ID
  "type": str,  # Question type
  "question": { 
    "stem": str,  # Question text
    "choices": [{"label": str, "text": str}, ...]  # Choices (for MCQ)
  }, 
  "answerKey": str,  # Correct answer

  # Miscellaneous fields...
}
```
This format is similar to the one used by the [AI2 Reasoning Challenge (ARC) dataset](https://leaderboard.allenai.org/arc/submissions/get-started). The miscellaneous fields can include any additional information about the question, such as the subject (e.g., "math"). `KCluster` will ignore these fields when generating KCs, but will copy them to the output. 

> 🤔 If you plan to upload the KC model produced by `KCluster` to DataShop, this is a good place to include a key that helps you map the question back to the student-step file (more on this later). 


A concrete example of a MCQ is as follows:
```python
{
  "id": "sciqa-2327",
  "type": "Multiple Choice",
  "question": {
    "stem": "Which is more flexible?",
    "choices": [
      {"label": "a", "text": "diamond"},
      {"label": "b", "text": "wool sweater"}
    ]
  },
  "answerKey": "b",

  # Miscellaneous fields
  "skill": "Compare properties of materials",
  "subject": "natural science",
  "topic": "physics",
  "category": "Materials",
  "grade": "grade3"
}
```

If the question is **not** a multiple-choice question, please do not include `choices` in the `question` field. The `answerKey` field should be the correct answer for the question. A concrete example of a non-MCQ is as follows:
```python
{
  "id": "spacing-1161-fib-5",
  "type": "Fill in the Blank",
  "question": {
    "stem": "When forces are _____, an object’s motion does not change."
  },
  "answerKey": "balanced",

  # Miscellaneous fields
  "LO": "I can compare balanced vs unbalanced forces",
  "Topic": "6.3 Force and Motion",
}
```

We include the datasets described in our paper: 
- `data/sciqa/sciqa-skill-10.jsonl` for a subset of the [ScienceQA](https://scienceqa.github.io/) dataset;
- `data/elearning/elearning22-mcq.jsonl` for the [E-learning 2022](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5426) dataset;
- `data/elearning/elearning23-mcq.jsonl` for the [E-learning 2023](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=5843) dataset.

You can use the provided datasets to test `KCluster` and see how it works; feel free to add your own datasets in the same format. We use the environment variable `$DATA_PATH` to denote the path to your `.jsonl` file (e.g., `data/sciqa-skill-10.jsonl`).

### 2. Extract concepts and question embeddings
As described in our paper, `KCluster` is compared to three baseline methods: `Concept`, `Concept-emb` and `Question-emb`. In addition, `KCluster` also uses the concept labels generated by `Concept` to name the KCs. To run `KCluster`, you need to extract the concepts first. 

You can run the script `experiments/run_concept.py` to extract concepts and question embeddings. To extract *just* the concepts (to be used with `KCluster` but not `Concept-emb` or `Question-emb`), you can run the following command:
```bash
python -m experiments.run_concept --llm_path "$LLM_PATH" \
                                  --data_path "$DATA_PATH"  \
                                  --output_dir "$OUTPUT_DIR" \
                                  --batch_size "$BATCH_SZ" \
                                  --num_beams 5 --length_penalty -0.1 \
                                  --pad_to_multiple_of 8
```
If you do not specify an `$OUTPUT_DIR`, the output will be saved in the project directory, under `results/concept/$(date)`(something like `results/concept/Wed_Mar_26_10-17-41_2025`). The variable `$BATCH_SZ` is the batch size for the LLM inference. The `--num_beams` and `--length_penalty` arguments are used to control the generation of concept labels; you can adjust them according to your needs. Setting `--pad_to_multiple_of` to a multiple of 8 may [accelerate the inference speed](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings.pad_to_multiple_of) on some GPUs with tensor cores but may consume more memory; the default is `None` (no padding) if you omit this argument.

The main output of this step is a CSV file named `*-concept.csv` in the output directory (where `*` is the name of your input file). The last column of the CSV file, `KC`, contains the concept labels generated by `Concept`; all other columns are copied from your input file. We call the directory that contains such concept CSV file `$CONCEPT_DIR`. 


### 3. Calculate question congruity
As described in our paper, `KCluster` uses `Phi-2` to measure question congruity, which a clustering algorithm can use to group questions into KCs. Since question congruity is mathematically similar to the pointwise mutual information (PMI) between two words, we abbreviate it as `PMI`. To calculate the PMI between questions, you can run the script `experiments/run_pmi.py`:
```bash
TIME="$(date)" # Get the current time before you spawn multiple GPUs
python -m experiments.run_pmi --llm_path "$LLM_PATH" \
                              --data_path "$DATA_PATH" \
                              --batch_size "$BATCH_SZ" \
                              --output_dir "results/pmi/$TIME" \
                              --pad_to_multiple_of 8 --num_workers 2
```
Since this script uses multiple GPUs (if available), you should set a fixed `--output_dir` to ensure that all PMI values are saved to the same directory. In the example above, we get the current time before we spawn multiple GPUs and use it to name our output directory—this way, we can ensure that all GPU processes write to the same directory. 

The PMI values are saved in a set of `.pt` files that are not meant to be human-readable. But we have another script that can extract KCs from these PMI values. Let's denote the directory that contains the PMI files as `$PMI_DIR`.

### 4. Extract KCs

So far we have generated the concepts for each question, which are saved to `$CONCEPT_DIR`, and calculated the question congruity (PMI values) for each question pair, which are saved to `$PMI_DIR`. Now we can run the script `evaluation/build_kc.py` to create KCs using `KCluster`:
```bash
python -m evaluation.build_kc --concept_dir "$CONCEPT_DIR" \
                              --pmi_dir "$PMI_DIR" \
                              --output_dir "$OUTPUT_DIR"
```
If `--output_dir` is omitted, the output will be saved to `results/kc/$(date)`. You can run this script on an ordinary laptop without a GPU, as it does not require any LLM inference.

This script will scan both directories and extract useful information to create KCs. As it runs, you should see some status updates such as:
```
*** Building KCs for KCluster-PMI ***
Affinity Propagation completed in 27 iterations and created 100 clusters
*** Finished with 92 KCs ***
```

At completion, you should find at least two CSV files in the output directory:
- `concept-kc.csv`: This is a copy of the concept CSV file, `*-concept.csv`, which is already a KC model.
- `pmi-kc.csv`: This is the KC model generated by `KCluster` using the PMI values. 

If you also applied `Concept-emb` or `Question-emb`, you should also find the corresponding KC models in the output directory (e.g., `concept-cosine-kc.csv` and `question-cosine-kc.csv`). The last column of each CSV file, `KC`, contains the KC labels generated by `KCluster` or the corresponding baseline method. 


### 5. Create DataShop-compatible KC models
[TODO]

## Citation
If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{wei2025kcluster,
  title={KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery},
  author={Wei, Yumou and Carvalho, Paulo and Stamper, John},
  booktitle={Proceedings of the 18th International Conference on Educational Data Mining (EDM)},
  month={July},
  year={2025},
  publisher={International Educational Data Mining Society},
  address={Palermo, Italy},
  url={https://arxiv.org/abs/2505.06469}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

