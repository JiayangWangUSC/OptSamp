#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiayangw@usc.edu
#SBATCH --account=jhaldar_118

module purge
module load gcc/8.3.0
module load cuda/11.2.0

python /project/jhaldar_118/jiayangw/OptSamp/uniform_training.py
