#!/usr/bin/env bash
#SBATCH -p batch
#SBATCH --job-name=xseciot-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=error_%j.txt
#SBATCH --chdir=/home/seth/Desktop/XSecIoT

set -euo pipefail

# Call your existing entrypoint
exec bash scripts/run_sim.sh