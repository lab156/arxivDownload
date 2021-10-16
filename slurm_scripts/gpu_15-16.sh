#!/bin/bash
#SBATCH --time=1-17:00:00
#SBATCH --job-name=15_16
#SBATCH --output=15_16.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p


cp $PROJECT/lstm_fix.sif $LOCAL
mkdir $LOCAL/promath
cp -r $PROJECT/promath/math{15..16} $LOCAL/promath
singularity run --nv --bind $LOCAL/promath:/opt/data_dir:rw $LOCAL/lstm_fix.sif /opt/data_dir/math{15..16}

# CREATE RESULTS DIRECTORY
res_dir=$PROJECT/inference_15_16
mkdir $res_dir
cp -r /tmp/rm_me_data/* $res_dir
mv $res_dir/classifying.log $res_dir/classifying_15_16.log
