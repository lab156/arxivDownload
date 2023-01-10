#!/bin/bash
#SBATCH --time=0-4:00:00
#SBATCH --job-name=ner_all2
#SBATCH --output=ner_all2.txt
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
OUTDIRNAME=ner_all_with_mp2
OLDPROJECT=/ocean/projects/dms200016p/lab232
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$OLDPROJECT:/opt/data_dir \
    $HOME/singul/runner.sif python3 embed/mp_inference_ner.py \
    --model /opt/data_dir/ner_search/trained_ner/lstm_ner/ner_Sep-29_15-37/exp_041 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/with_mp/math*/*.xml.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./ner_all2.txt $PROJECT/$OUTDIRNAME/trainer_logs
