#!/bin/bash
#SBATCH --time=0-10:00:00
#SBATCH --job-name=ner_search
#SBATCH --output=ner_search.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p


singularity run --nv $PROJECT/ner_train_units.sif --epochs 150 --experiments 10

# CREATE RESULTS DIRECTORY
results_dir=$PROJECT/ner_search
mkdir $results_dir
cp -r /tmp/trainer/* $results_dir

