#!/bin/bash
#SBATCH --job-name=visualize_qqp
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/qqp_visualize.out
#SBATCH --error=slurm_errors/qqp_visualize.err
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
CONFIG_FILE=${1:-"configs/benchmarks/qqp/qqp_visualize.json"}

echo "Starting QQP benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/visualize_qqp.py --config "$CONFIG_FILE"

echo "Visualize QQP script completed at $(date)"