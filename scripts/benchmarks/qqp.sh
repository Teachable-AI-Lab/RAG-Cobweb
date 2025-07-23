#!/bin/bash
#SBATCH --job-name=benchmark_qqp
#SBATCH --time=02:30:00
#SBATCH --mem=16G
#SBATCH --output=/nethome/$USER/flash/slurm_outputs/qqp_benchmark.out
#SBATCH --error=/nethome/$USER/flash/slurm_errors/qqp_benchmark.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd /nethome/$USER/flash/RAG-Cobweb
export PYTHONPATH=$(pwd)

# Default arguments - can be overridden by command line arguments
CONFIG_FILE=${1:-"configs/benchmarks/qqp_default.json"}

echo "Starting QQP benchmark at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/qqp_dataset.py --config "$CONFIG_FILE"

echo "QQP benchmark completed at $(date)"