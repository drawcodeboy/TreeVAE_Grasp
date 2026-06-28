#!/bin/bash
#SBATCH --job-name=hograspnet_download
#SBATCH --output=/workspace/lab_intern/KDW/HOGraspNet/logs/slurm_%j.out
#SBATCH --error=/workspace/lab_intern/KDW/HOGraspNet/logs/slurm_%j.err
#SBATCH --partition=intern
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=intern_qos
#SBATCH --time=2-00:00:00  

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate kdw_grasp

cd /workspace/lab_intern/KDW/HOGraspNet
mkdir -p ./logs

export HOG_DIR=/workspace/lab_intern/KDW/HOGraspNet

export PYTHONUNBUFFERED=1
python -u scripts/download_data.py --type=3 --subject=all --objModel=True --output_folder=/workspace/lab_intern/KDW/HOGraspNet/data

echo "Job ID: $SLURM_JOB_ID" >> logs/job_info.txt
echo "Node: $SLURM_NODELIST" >> logs/job_info.txt
echo "GPU: $CUDA_VISIBLE_DEVICES" >> logs/job_info.txt
echo "Completed: $(date)" >> logs/job_info.txt