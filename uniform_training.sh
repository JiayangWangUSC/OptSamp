#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiayangw@usc.edu
#SBATCH --account=jhaldar_118

module purge
module load gcc/8.3.0
module load cuda/10.1.243

run /project/jhaldar_118/jiayangw/OptSamp/uniform_training.py
