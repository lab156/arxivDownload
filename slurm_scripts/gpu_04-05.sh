#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --job-name=04_05
#SBATCH --output=04_05.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p


cp $PROJECT/lstm.sif $LOCAL
mkdir $LOCAL/promath
cp -r $PROJECT/promath/math{04,05} $LOCAL/promath
singularity run --nv --bind $LOCAL/promath:/opt/data_dir:rw $LOCAL/lstm.sif /opt/data_dir/math{04,05}

# CREATE RESULTS DIRECTORY
res_dir=$PROJECT/inference_04_05
mkdir $res_dir
cp -r /tmp/rm_me_data/* $res_dir
mv $res_dir/classifying.log $res_dir/classifying_04_05.log
