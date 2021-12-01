#!/bin/bash
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --mem=40gb                          # Job memory request --mem=20gb
#SBATCH --time=120:00:00                    # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:1                        #
#SBATCH --exclude=gnodek1,gnodec1,gnodee8   # gnodee6,gnodee1

#SBATCH -o /users/xuji/abstract/out/slurm/slurm_%j.out
#SBATCH -e /users/xuji/abstract/out/slurm/slurm_%j.out


cd /users/xuji/abstract
source ../env_python3_3/bin/activate
echo $1
eval $1