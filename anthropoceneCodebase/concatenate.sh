#!/bin/sh
#SBATCH -A p32234
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:30:00
#SBATCH --mem=16G

module purge
module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate condascreenshots
python /home/ysc4337/aerith/anthropocene-reconcile/anthropocene-ML/anthropoceneCodebase/concatenateCaptions.py