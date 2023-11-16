#!/bin/bash
#SBATCH --time=1-06:00:00
#SBATCH --job-name=class_ft
#SBATCH --output=class_ft-%j.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU
#SBATCH --gpus=8

# To be run in interact mode potentially with multiple gpus
OUTDIRNAME="finetune/class-"$(date "+%Y-%m-%d_%H%M")

export TRANSFORMERS_CACHE=$PROJECT/hfcache
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerPytorchTransHF3.sif torchrun \
    LLMs/mirror_pytorch_finetune_class_HF.py \
    --savedir /opt/data_dir/$OUTDIRNAME/model \
    --configpath /opt/arxivDownload/rmme_config.toml

#mkdir $PROJECT/$OUTDIRNAME/
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_ft-$SLURM_JOB_ID.txt $PROJECT/$OUTDIRNAME/trainer_logs
