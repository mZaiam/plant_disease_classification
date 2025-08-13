#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --time=48:00:00
#SBATCH --nodelist=work1

source ~/.bashrc
conda activate ai
module load cuda/12.2
python -u $1 > $2 
