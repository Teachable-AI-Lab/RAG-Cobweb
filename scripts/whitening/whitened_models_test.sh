#!/bin/bash
#SBATCH --job-name=whitened_models_test
#SBATCH --time=04:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/whitened_models_test.out
#SBATCH --error=slurm_errors/whitened_models_test.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/RAG-Cobweb
export PYTHONPATH=$(pwd)

echo "Starting Whitened Models Embeddings Test at $(date)"

srun python src/whitening/whitened_models_test.py

echo "Whitened Models Embeddings Test script completed at $(date)"