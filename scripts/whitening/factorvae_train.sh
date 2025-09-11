#!/bin/bash
#SBATCH --job-name=factorvae_train
#SBATCH --time=24:00:00
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

DATASET=${1:-"qqp"}

echo "Starting Training of FactorVAE at $(date)"
echo "Using dataset: $DATASET"

srun python src/whitening/factorvae_train.py --task "$DATASET" --epochs 20 --batch-size 256 --max-embed-samples 10000

echo "Finished Training of FactorVAE at $(date)"