#!/bin/bash
#SBATCH --job-name=train_whitening
#SBATCH --time=00:00:00
#SBATCH --output=/nethome/ksingara3/slurm_outputs/train_whitening.out
#SBATCH --error=/nethome/ksingara3/slurm_errors/train_whitening.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd /nethome/ksingara3/RAG-Cobweb
export PYTHONPATH=$(pwd)

srun python src/whitening/whitening_finetune.py