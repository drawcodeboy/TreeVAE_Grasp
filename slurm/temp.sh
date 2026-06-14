#!/bin/bash
#SBATCH --job-name=debug_treevae
#SBATCH --output=/workspace/lab_intern/KDW/treevae/logs/slurm_%j.out
#SBATCH --error=/workspace/lab_intern/KDW/treevae/logs/slurm_%j.err
#SBATCH --partition=intern
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --qos=intern_qos
#SBATCH --time=2-00:00:00  

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate treevae

cd /workspace/lab_intern/KDW/treevae
mkdir -p ./logs

python -u main.py --config dexgraspnet

echo "Job ID: $SLURM_JOB_ID" >> logs/job_info.txt
echo "Node: $SLURM_NODELIST" >> logs/job_info.txt
echo "Completed: $(date)" >> logs/job_info.txt