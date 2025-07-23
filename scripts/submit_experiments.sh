#!/bin/bash
#SBATCH --job-name=run_experiments
#SBATCH --time=00:30:00
#SBATCH --output=/nethome/agupta886/flash/slurm_outputs/run_experiments.out
#SBATCH --error=/nethome/agupta886/flash/slurm_errors/run_experiments.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd /nethome/agupta886/flash/RAG-Cobweb
export PYTHONPATH=$(pwd)

echo "Starting experiment runner at $(date)"
echo "Arguments: $@"

# Run the master script with all arguments passed to this script
python scripts/run_experiments.py "$@"

echo "Experiment runner completed at $(date)"
