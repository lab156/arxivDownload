#!/bin/bash
#SBATCH --time=0-10:00:00
#SBATCH --job-name=ner_all
#SBATCH --output=ner_all.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
##SBATCH --partition=GPU
##SBATCH --gpus=8

#source $HOME/env1/bin/activate
#module load AI/anaconda3-tf2.2020.11
#source activate $AI_ENV

export TF_CUDNN_RESET_RND_GEN_STATE=1
OUTDIRNAME=ner_all_from_with_mp
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $HOME/singul/runner.sif python3 embed/inference_ner.py \
    --model /opt/data_dir/ner_search/trained_ner/lstm_ner/ner_Sep-29_15-37/exp_041 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/with_mp/math*/*.xml.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./ner_all.txt $PROJECT/$OUTDIRNAME/trainer_logs
