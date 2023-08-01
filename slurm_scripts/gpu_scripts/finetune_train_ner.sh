#!/bin/bash
#SBATCH --time=0-2:00:00
#SBATCH --job-name=finetuning
#SBATCH --output=finetune_ner.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
##SBATCH --partition=GPU
##SBATCH --gpus=8


#export TF_CUDNN_RESET_RND_GEN_STATE=1
OUTDIRNAME="finetune_ner/ner-"$(date "+%Y-%m-%d_%H%M")
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerNERNew.sif python3 LLMs/fine_tune_ner_HF.py \
    --configpath /opt/arxivDownload/config.toml

mkdir -p $PROJECT/$OUTDIRNAME/
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME
cp ./finetune_ner.txt $PROJECT/$OUTDIRNAME/trainer_logs.txt

