#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --job-name=c_bb_90s
#SBATCH --output=class_bb_90s.txt
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
OUTDIRNAME=best_bridges
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $HOME/singul/runner.sif python3 embed/mp_classify.py \
    --model /opt/data_dir/trained_models/lstm_classifier/Bridges-2-best/exp_008 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/promath/math9*/*.tar.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_bb_90s.txt $PROJECT/$OUTDIRNAME/trainer_logs
