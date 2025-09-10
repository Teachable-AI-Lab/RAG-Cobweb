#!/bin/bash
#SBATCH --job-name=factorvae_train
#SBATCH --time=04:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/factorvae_train.out
#SBATCH --error=slurm_errors/factorvae_train.err
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

srun python src/whitening/factorvae_train.py

echo "Whitened Models Embeddings Test script completed at $(date)"