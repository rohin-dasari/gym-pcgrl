#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=test.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=rd2893@nyu.edu
#SBATCH --output=rl_runs/test.out
