#!/bin/bash 

# env name: maze_environment_narrow_ppo_shared_weights_checkerboard.out

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --job-name=maze_environment_narrow_ppo_shared_weights_checkerboard.out
#SBATCH --time=168:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=rd2893@nyu.edu
#SBATCH --output=/scratch/rd2893/maze_environment_narrow_ppo_shared_weights_checkerboard.out

source /scratch/rd2893/miniconda3/bin/activate pcgrl
#python qmix_test.py
python main.py -c=/home/rd2893/gym-pcgrl/configs/binary_actions_maze_narrow.yaml

