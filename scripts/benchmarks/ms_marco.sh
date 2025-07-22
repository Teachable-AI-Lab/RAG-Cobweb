#!/bin/bash
#SBATCH --job-name=benchmark_ms_marco
#SBATCH --time=02:30:00
#SBATCH --output=/nethome/agupta886/flash/slurm_outputs/ms_marco.out
#SBATCH --error=/nethome/agupta886/flash/slurm_errors/ms_marco.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd /nethome/agupta886/flash/RAG-Cobweb
export PYTHONPATH=$(pwd)

srun python src/benchmarks/ms_marco_dataset.py