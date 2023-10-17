#!/bin/bash

# To be run in interact mode potentially with multiple gpus

singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerTransHF.sif python3 embed/mirror_finetune_class_HF.py \
    --savedir /opt/data_dir/result/model \
    --configpath /opt/arxivDownload/rmme_config.toml
