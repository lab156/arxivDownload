#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --job-name=ner_test
#SBATCH --output=ner_test.txt
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
OUTDIRNAME=ner_all
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerNERNew.sif python3 LLMs/mp_infer_HFTrans_ner.py \
    --model /opt/data_dir/finetune_ner/ner-2023-08-02_1858/trainer/trans_HF_ner/ner_Aug-02_18-58/ \
    --out /tmp/infer_ner  \
    --mine /opt/data_dir/HFTrans_infer_all/math0{0,1}/*.xml.gz \
    --senttok /opt/data_dir/punkt_params.pickle 

cp -r /tmp/infer_ner $PROJECT/$OUTDIRNAME/infer_ner
cp ./ner_test.txt $PROJECT/$OUTDIRNAME/infer_ner
