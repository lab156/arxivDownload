#!/bin/bash
#SBATCH --time=0-3:00:00
#SBATCH --job-name=class_task
#SBATCH --output=class_task.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
##SBATCH --partition=GPU-shared
##SBATCH --gpus=1
#SBATCH --partition=GPU
#SBATCH --gpus=8

#source $HOME/env1/bin/activate
#module load AI/anaconda3-tf2.2020.11
#source activate $AI_ENV

singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,\
    $PROJECT:/opt/data_dir \
    $HOME/singul/runner.sif python3 embed/mp_classify.py \
    --model /opt/data_dir/trained_models/lstm_classifier/lstm_Aug-19_17-22 \
    --out $PROJECT/with_mp_classify \
    --mine /opt/data_dir/promath/math00/*.tar.gz
cp -r /tmp/trainer $PROJECT/with_mp_classify/trainer_logs
