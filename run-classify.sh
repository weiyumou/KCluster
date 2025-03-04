#!/bin/bash

#SBATCH --job-name=classify
#SBATCH --account=yumouwei0
#SBATCH --partition=spgpu
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module load python3.11-anaconda/2024.02 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

# Run code
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate llm-new
cd /gpfs/accounts/yumouwei_root/yumouwei0/yumouwei/KCluster/ || exit
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128


LLM_PATH="/home/yumouwei/turbo/llm/phi-2"
OUTPUT_DIR="results/classify/$(date)"

DATA_PATH="data/elearning/elearning-mcq.jsonl"
LO_PATH="data/datashop/ds5843-elearning/v1-prompt.txt"
LO_TYPE="actions"

srun python -m experiments.classify --llm_path "$LLM_PATH" --data_path "$DATA_PATH" --lo_path "$LO_PATH" \
                                    --lo_type "$LO_TYPE" --batch_size 80  --output_dir "$OUTPUT_DIR"
