#!/bin/bash
#SBATCH --job-name=multiocr
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=360GB
#SBATCH --time=72:00:00
#SBATCH --account=def-kshook
#SBATCH --cpus-per-task=16

#SBATCH -o ./output_igbo.log



# Modify the following lines according to your setup process
export ENVDIR=venv
source $ENVDIR/bin/activate

python3 -c "import torch; torch.cuda.empty_cache()"

nohup python3 main.py --lang igbo
