#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --job-name=small_fix
#SBATCH --output=small_fix.txt
#SBATCH --mail-user=lab232@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
##SBATCH --partition=GPU
##SBATCH --gpus=8

# Change Parameters and Variables
# --time
# --job-name
# --output (change also in the last line)
# --OUTDIRNAME
# maybe: --mine


export TF_CUDNN_RESET_RND_GEN_STATE=1
OUTDIRNAME=infer_2nd_half
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerTransHF.sif python3 LLMs/mp_HF_classify_inference.py \
    --model /ocean/projects/mth220001p/lab232/finetune/class-2023-08-07_1327 \
    --out $PROJECT/$OUTDIRNAME \
    --mine /opt/data_dir/promath/math09/0901_003.tar.gz
cp -r /tmp/trainer $PROJECT/$OUTDIRNAME/trainer_logs
cp ./small_fix.txt $PROJECT/$OUTDIRNAME/trainer_logs
