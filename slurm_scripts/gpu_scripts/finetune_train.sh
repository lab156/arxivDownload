#!/bin/bash
#SBATCH --time=1-10:00:00
#SBATCH --job-name=ft-%j
#SBATCH --output=finetune-%j.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
##SBATCH --partition=GPU
##SBATCH --gpus=8


#export TF_CUDNN_RESET_RND_GEN_STATE=1
OUTDIRNAME=finetune
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerTransHF.sif python3 embed/fine_tune_classify_HF.py \
    --model /opt/data_dir/finetuned_models/model_start \

cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_bb_2nd_half.txt $PROJECT/$OUTDIRNAME/trainer_logs