#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --account=yumouwei0
#SBATCH --partition=spgpu
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module load python3.11-anaconda/2024.02 cuda/12.6.3 cudnn/12.6-v9.6.0
module list

# Run code
source /sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate llm
cd /gpfs/accounts/yumouwei_root/yumouwei0/yumouwei/KCluster/ || exit


echo "$(date) $(hostname)"
export JUPYTER_RUNTIME_DIR="./jupyter/runtime"

jupyter lab --no-browser --port 9090 --notebook-dir=.
