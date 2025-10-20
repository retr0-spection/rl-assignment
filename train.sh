#!/bin/bash
#SBATCH --job-name=RL-DQN
#SBATCH --output=/home-mscluster/onailana/rl-assignment/result.txt
#SBATCH --partition=bigbatch
#SBATCH --nodes=1

source ~/.bashrc
conda env create -f environment.yml
conda init bash
conda activate crafter_env
pip install -r requirements.txt

python train.py
