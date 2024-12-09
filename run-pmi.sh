#!/bin/bash

#SBATCH --job-name=pmi
#SBATCH --account=yumouwei0
#SBATCH --partition=spgpu
#SBATCH --time=14-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2gb
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module load python3.11-anaconda/2024.02 cuda/12.1.1 cudnn/12.1-v8.9.0
module list

# Run code
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate llm-new
cd /gpfs/accounts/yumouwei_root/yumouwei0/yumouwei/KCluster/ || exit
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128


TIME="$(date)"
LLM_PATH="/home/yumouwei/turbo/llm/phi-2"

# spacing
DATA_PATH="data/spacing/spacing-all.jsonl"
BATCH_SZ=128

# sciqa
DATA_PATH="data/sciqa/sciqa-skill-10.jsonl"
BATCH_SZ=96

# elearning
DATA_PATH="data/elearning/elearning-mcq.jsonl"
BATCH_SZ=96

srun python -m experiments.run_pmi --llm_path "$LLM_PATH" --data_path "$DATA_PATH" \
                                   --batch_size $BATCH_SZ --output_dir "results/pmi/$TIME"
