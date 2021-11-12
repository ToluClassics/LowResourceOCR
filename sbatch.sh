#!/bin/bash
#SBATCH --job-name=multiocr
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --time=24:00:00
#SBATCH --account=def-kshook
#SBATCH --cpus-per-task=8

#SBATCH -o ./output.log


# Modify the following lines according to your setup process
ENVDIR=venv
source $ENVDIR/bin/activate

nohup python3 main.py
