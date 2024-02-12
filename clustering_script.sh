#!/bin/sh
#SBATCH -A p32234
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH -N 1
#SBATCH -n 52
#SBATCH -t 1:00:00
#SBATCH --mem=128G

source /home/ysc4337/projects/anthropocene_ML/anthropocene/bin/activate
python /home/ysc4337/projects/anthropocene_ML/bert_topic.py
