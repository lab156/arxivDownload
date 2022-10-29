#!/bin/bash
#SBATCH --time=0-20:00:00
#SBATCH --job-name=cells_search
#SBATCH --output=cells_search_%j.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p


#singularity run --nv $PROJECT/lstm_cells.sif --epochs 15 --experiments 10
singularity run --nv --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir $HOME/singul/runner.sif python3 embed/train_lstm3.py --epochs 15 --experiments 10 \
    --startat 7


# CREATE RESULTS DIRECTORY
#results_dir=$PROJECT/cells_search2
#mkdir $results_dir
#cp -r /tmp/trainer/* $results_dir

