#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:01:00
#SBATCH --output=/flash/slurm_outputs/test.out
#SBATCH --error=/flash/slurm_errors/test.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

echo "$HOME"
echo "$PWD"