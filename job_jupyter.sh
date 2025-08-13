#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p batch-AMD
#SBATCH --time=12:00:00

source ~/.bashrc

conda activate ai

ipaddress=$(ip addr | grep 172 | awk 'NR==1{print $2}' | sed 's!/23!!g' | sed 's!/0!!g')
echo $ipaddress

jupyter-notebook --ip=$ipaddress


