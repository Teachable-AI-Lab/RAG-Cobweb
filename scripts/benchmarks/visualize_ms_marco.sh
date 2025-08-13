#!/bin/bash
#SBATCH --job-name=visualize_ms_marco
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/ms_marco_visualize.out
#SBATCH --error=slurm_errors/ms_marco_visualize.err
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
CONFIG_FILE=${1:-"configs/benchmarks/ms_marco/ms_marco_visualize.json"}

echo "Starting MS Marco benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/visualize_ms_marco.py --config "$CONFIG_FILE"

echo "Visualize MS Marco script completed at $(date)"