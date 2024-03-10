#!/bin/sh
#SBATCH -A p32234
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH -N 1
#SBATCH -n 52
#SBATCH -t 4:00:00
#SBATCH --mem=128G

module purge
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate knowledge
python /home/ysc4337/aerith/anthropocene-reconcile/anthropocene-ML/anthropoceneCodebase/screenshotData.py