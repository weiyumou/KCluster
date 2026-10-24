from transformers.utils import logging

from kcluster.engine.local import DTYPES, LargeLangModel, LogProbScorer, load_adapter, score_pairs
from kcluster.io import figures
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

    # The questions, and where their figures are (shown to a model that takes images)
    questions = load_questions(args.data_path)
    figure_root = figures.figure_root(questions, args.data_path, args.figure_dir, args.ignore_figures)
    adapter = load_adapter(args.llm_path, figure_root)

    # Load tokenizer
    tokenizer = LargeLangModel.load_tokenizer(args.llm_path)

    # Load model
    model = LogProbScorer(args.llm_path, trust_remote_code=True, dtype=DTYPES[args.dtype])

    # Compute PMI
    score_pairs(model, tokenizer, PairQuestion(questions, figure_root), shard_dir, batch_size=args.batch_size,
                num_workers=args.num_workers, pad_to_multiple_of=args.pad_to_multiple_of, adapter=adapter,
                strategy="ddp")

    save_args(result_dir, "pmi", dataset_name(args.data_path), vars(args))


def add_arguments(parser):
    parser.add_argument("--llm_path", required=True, type=str, help="Path to a downloaded LLM")
    parser.add_argument("--data_path", required=True, type=str, help="Path to a jsonl file of questions")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of pairs to score in a batch (a 262k-token vocabulary such as Gemma 4's wants "
                             "a small one)")
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES),
                        help="Precision to load the model in (Gemma wants bfloat16)")
    figures.add_arguments(parser)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="The result directory (default: --run_dir or a fresh timestamped folder)")
    parser.add_argument("--run_dir", default=None, type=str,
                        help="Result folder shared by every step of this run (env: KCLUSTER_RUN_DIR)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--pad_to_multiple_of", type=int, default=None, help="Pad to multiple of")
