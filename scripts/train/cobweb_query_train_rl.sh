#!/bin/bash
#SBATCH --job-name=cobweb_query_train_rl
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm_outputs/cobweb_query_train_rl.out
#SBATCH --error=slurm_errors/cobweb_query_train_rl.err
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

echo "Starting Reinforcement-Learning-Based Training of Query Encoder by Cobweb at $(date)"

srun python src/training/cobweb_query_train_rl.py

echo "Finished Reinforcement-Learning-Based Training of Query Encoder by Cobweb at $(date)"