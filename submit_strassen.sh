#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=strassen
#SBATCH --output=logs/strassen_%j.out
#SBATCH --error=logs/strassen_%j.err

source /scratch/gpfs/ashwinee/activate.sh neox
cd /scratch/gpfs/ashwinee/strassen-triton
python profiling.py