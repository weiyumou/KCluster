#!/bin/bash

#SBATCH --job-name=concept
#SBATCH --account=yumouwei0
#SBATCH --partition=spgpu
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4gb
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module load python3.11-anaconda/2024.02 cuda/12.6.3 cudnn/12.6-v9.6.0
module list

# Run code
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate llm
cd /gpfs/accounts/yumouwei_root/yumouwei0/yumouwei/KCluster/ || exit
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128


LLM_PATH="/home/yumouwei/turbo/llm/phi-2"
SENT_PATH="/home/yumouwei/turbo/llm/all-mpnet-base-v2"

# sciqa
DATA_PATH="data/sciqa/sciqa-skill-10.jsonl"
BATCH_SZ=16

# elearning 22
DATA_PATH="data/elearning/elearning22-mcq.jsonl"
BATCH_SZ=16

# elearning 23
DATA_PATH="data/elearning/elearning23-mcq.jsonl"
BATCH_SZ=16

python -m experiments.run_concept --llm_path "$LLM_PATH" --data_path "$DATA_PATH" \
                                  --sent_path "$SENT_PATH" --q_embeds \
                                  --batch_size "$BATCH_SZ" --num_beams 5 --length_penalty -0.1 \
                                  --pad_to_multiple_of 8
