#!/bin/bash
#SBATCH --job-name=hograspnet_pointcloud_contactmap_prediction
#SBATCH --output=/workspace/dwkwon/treevae/logs/slurm_%j.out
#SBATCH --error=/workspace/dwkwon/treevae/logs/slurm_%j.err
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00  

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate treevae

cd /workspace/dwkwon/treevae
mkdir -p ./logs

export WANDB_API_KEY=

python -u main.py --config_name=hograspnet_pointcloud_contactmap_prediction

echo "Job ID: $SLURM_JOB_ID" >> logs/job_info.txt
echo "Node: $SLURM_NODELIST" >> logs/job_info.txt
echo "Completed: $(date)" >> logs/job_info.txt