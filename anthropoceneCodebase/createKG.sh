#!/bin/sh
#SBATCH -A p32234
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 1:00:00

module purge
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate knowledge
python /home/ysc4337/aerith/anthropocene-reconcile/anthropocene-ML/anthropoceneCodebase/createKG.py