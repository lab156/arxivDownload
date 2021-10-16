#!/bin/bash
#SBATCH --time=0-4:00:00
#SBATCH --job-name=the90s
#SBATCH --output=the90s.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p


cp $PROJECT/lstm.sif $LOCAL
mkdir $LOCAL/promath
cp -r $PROJECT/promath/math9* $LOCAL/promath
singularity run --nv --bind $LOCAL/promath:/opt/data_dir:rw $LOCAL/lstm.sif /opt/data_dir/math9*

# CREATE RESULTS DIRECTORY
res_dir=$PROJECT/inference_the90s
mkdir $res_dir
cp -r /tmp/rm_me_data/* $res_dir
mv $res_dir/classifying.log $res_dir/classifying_the90s.log
