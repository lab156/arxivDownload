#!/bin/bash
#SBATCH --time=0-6:00:00
#SBATCH --job-name=class_task
#SBATCH --output=class_task.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --gres=gpu:1

source $HOME/env1/bin/activate
module load AI/anaconda3-tf2.2020.11
source activate $AI_ENV

cd $HOME/arxivDownload/embed
python3 train_lstm.py
