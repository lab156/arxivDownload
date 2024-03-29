#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --job-name=class_8bit
#SBATCH --output=class_ft-%j.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:4

# To be run in interact mode potentially with multiple gpus
OUTDIRNAME="finetune/class-"$(date "+%Y-%m-%d_%H%M")

export TRANSFORMERS_CACHE=$PROJECT/hfcache
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerPytorchTransHF4.sif torchrun \
    LLMs/mirror_pytorch_finetune_class_HF.py \
    --savedir /opt/data_dir/$OUTDIRNAME/model \
    --configpath /opt/arxivDownload/rmme_config.toml \
    --load8bit

#mkdir $PROJECT/$OUTDIRNAME/
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_ft-$SLURM_JOB_ID.txt $PROJECT/$OUTDIRNAME/trainer_logs
