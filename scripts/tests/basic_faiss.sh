#!/bin/bash
#SBATCH --job-name=basic_faiss_test
#SBATCH --time=00:03:00
#SBATCH --output=/nethome/agupta886/flash/slurm_outputs/basic_faiss_test.out
#SBATCH --error=/nethome/agupta886/flash/slurm_errors/basic_faiss_test.err
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

srun python tests/basic_faiss_test.py