#!/bin/bash
#SBATCH --time=1-11:00:00
#SBATCH --job-name=13_14
#SBATCH --output=13_14.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p


cp $PROJECT/lstm_fix.sif $LOCAL
mkdir $LOCAL/promath
cp -r $PROJECT/promath/math{13..14} $LOCAL/promath
singularity run --nv --bind $LOCAL/promath:/opt/data_dir:rw $LOCAL/lstm_fix.sif /opt/data_dir/math{13..14}

# CREATE RESULTS DIRECTORY
res_dir=$PROJECT/inference_13_14
mkdir $res_dir
cp -r /tmp/rm_me_data/* $res_dir
mv $res_dir/classifying.log $res_dir/classifying_13_14.log
