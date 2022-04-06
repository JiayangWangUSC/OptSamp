#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiayangw@usc.edu
#SBATCH --account=jhaldar_118

module purge
module load matlab

matlab -batch  "noiselevel=0.2;TV_gradient.m;quit"