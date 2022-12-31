#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --job-name=c_15-17
#SBATCH --output=class_15-17.txt
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
OUTDIRNAME=with_mp_15-17
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $HOME/singul/runner.sif python3 embed/mp_classify.py \
    --model /opt/data_dir/trained_models/lstm_classifier/lstm_Aug-19_17-22 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/promath/math{15,16,17}/*.tar.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_15-17.txt $PROJECT/$OUTDIRNAME/trainer_logs
