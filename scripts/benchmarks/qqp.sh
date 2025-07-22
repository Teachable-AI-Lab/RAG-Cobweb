#!/bin/bash
#SBATCH --job-name=benchmark_qqp
#SBATCH --time=05:00:00
#SBATCH --output=/nethome/ksingara3/flash/slurm_outputs/qqp_benchmark_pcaica.out
#SBATCH --error=/nethome/ksingara3/flash/slurm_errors/qqp_benchmark_pcaica.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd /nethome/ksingara3/flash/RAG-Cobweb
export PYTHONPATH=$(pwd)

srun python src/benchmarks/qqp_dataset.py