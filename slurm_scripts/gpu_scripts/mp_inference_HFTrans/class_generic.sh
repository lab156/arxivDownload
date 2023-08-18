#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --job-name=c_test-mp
#SBATCH --output=class_test-mp.txt
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
OUTDIRNAME=HF_test
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerTransHF.sif python3 LLMs/mp_HF_classify_inference.py \
    --model /ocean/projects/mth220001p/lab232/finetune/class-2023-08-07_1327 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/promath/math99/*.tar.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./class_test-mp.txt $PROJECT/$OUTDIRNAME/trainer_logs
