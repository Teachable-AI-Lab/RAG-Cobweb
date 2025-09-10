#!/bin/bash
#SBATCH --job-name=whitening_vicreg
#SBATCH --time=04:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/whitening_vicreg.out
#SBATCH --error=slurm_errors/whitening_vicreg.err
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

echo "Starting Whitening Training with VICREG at $(date)"

srun python src/whitening/whitening_vicreg.py

echo "Whitening VICReg script completed at $(date)"