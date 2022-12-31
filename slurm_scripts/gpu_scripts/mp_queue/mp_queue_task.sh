#!/bin/bash
#SBATCH --time=0-3:00:00
#SBATCH --job-name=class_lock
#SBATCH --output=class_00_lock.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
##SBATCH --partition=GPU-shared
##SBATCH --gpus=1
#SBATCH --partition=GPU
#SBATCH --gpus=8

#source $HOME/env1/bin/activate
#module load AI/anaconda3-tf2.2020.11
#source activate $AI_ENV

export TF_CUDNN_RESET_RND_GEN_STATE=1
OUTDIRNAME=with_mp_lock
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $HOME/singul/runner.sif python3 embed/mp_classify.py \
    --model /opt/data_dir/trained_models/lstm_classifier/lstm_Aug-19_17-22 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/promath/math00/*.tar.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_00_lock.txt $PROJECT/$OUTDIRNAME/trainer_logs
