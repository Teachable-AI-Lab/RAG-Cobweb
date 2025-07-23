#!/bin/bash
#SBATCH --job-name=benchmark_ms_marco
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --output=slurm_outputs/ms_marco.out
#SBATCH --error=slurm_errors/ms_marco.err
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

# Default arguments - can be overridden by command line arguments
CONFIG_FILE=${1:-"configs/benchmarks/ms_marco_default.json"}

echo "Starting MS Marco benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/ms_marco_dataset.py --config "$CONFIG_FILE"

echo "MS Marco benchmark completed at $(date)"