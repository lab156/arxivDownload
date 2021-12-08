#!/bin/bash
#SBATCH --time=0-1:00:00
#SBATCH --job-name=test_9697
#SBATCH --output=test_9697.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

##SBATCH --partition=GPU
##SBATCH --gpus=8
##SBATCH -A tra210003p

#source $HOME/env1/bin/activate
#module load AI/anaconda3-tf2.2020.11
#source activate $AI_ENV


# Running the conv class
#touch $HOME/class_batch_job_acc.log
#singularity run --nv $HOME/conv.sif
#cp /tmp/rm_me_data/classifying.log $HOME/class_conv_job.log

cp $PROJECT/lstm.sif $LOCAL
mkdir $LOCAL/promath
cp -r $PROJECT/promath/{math96,math97} $LOCAL/promath
singularity run --nv --bind $LOCAL/promath:/opt/data_dir:rw $LOCAL/lstm.sif /opt/data_dir/{math96,math97}

# CREATE RESULTS DIRECTORY
res_dir=$PROJECT/test_9697
mkdir $res_dir
cp -r /tmp/rm_me_data/* $res_dir
