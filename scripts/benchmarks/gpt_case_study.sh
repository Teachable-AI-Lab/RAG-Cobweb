#!/bin/bash
#SBATCH --job-name=gpt_case_study
#SBATCH --time=03:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/gpt_case_study.out
#SBATCH --error=slurm_errors/gpt_case_study.err
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
CONFIG_FILE=${1:-"configs/benchmarks/gpt_case_study.json"}

echo "Starting GPT Case Study at $(date)"
echo "Using config: $CONFIG_FILE"

srun python src/benchmarks/gpt_case_study.py --config "$CONFIG_FILE"

echo "GPT Case Study completed at $(date)"