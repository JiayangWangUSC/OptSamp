#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --account=jhaldar_118

module purge
module load gcc/8.3.0
module load cuda/10.1.243
