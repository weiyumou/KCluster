import torch
from transformers.utils import logging

from kcluster.engine.local import LargeLangModel, LogProbScorer, score_pairs
from kcluster.io.jsonl import load_questions
from kcluster.paths import dataset_name, default_result_dir, pmi_raw_dir, prepare_output_dir, save_args
from kcluster.tasks.congruity import PairQuestion


def main(args):
    logging.set_verbosity(logging.ERROR)  # Suppress warnings from transformers

    # Resolve the per-dataset result folder (D10 layout); the raw score
    # shards are an intermediate, so they live under mat/pmi/raw/
    result_dir = args.output_dir or default_result_dir(args.run_dir)
    args.output_dir = result_dir = prepare_output_dir(result_dir)
    shard_dir = prepare_output_dir(pmi_raw_dir(result_dir))
    print(f"*** Writing results to {result_dir} ***")

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = LogProbScorer(args.llm_path, trust_remote_code=True, torch_dtype=torch.float16)

    # Compute PMI
    questions = load_questions(args.data_path)
    score_pairs(model, tokenizer, PairQuestion(questions), shard_dir, batch_size=args.batch_size,
                num_workers=args.num_workers, pad_to_multiple_of=args.pad_to_multiple_of, strategy="ddp")

    save_args(result_dir, "pmi", dataset_name(args.data_path), vars(args))


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of questions to process in a batch")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The result directory (default: --run_dir or a fresh timestamped folder)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")
