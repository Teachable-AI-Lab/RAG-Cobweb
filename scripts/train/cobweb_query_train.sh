#!/bin/bash
#SBATCH --job-name=cobweb_query_train
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/cobweb_query_train.out
#SBATCH --error=slurm_errors/cobweb_query_train.err
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

echo "Starting Training of Query Encoder by Cobweb Contrastive Learning at $(date)"

srun python src/training/cobweb_query_train.py

echo "Finished Training of Query Encoder by Cobweb Contrastive Learning at $(date)"