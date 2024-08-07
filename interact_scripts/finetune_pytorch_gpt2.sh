#!/bin/bash

# To be run in interact mode potentially with multiple gpus

export TRANSFORMERS_CACHE=$PROJECT/hfcache
singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerPytorchTransHF3.sif python3 \
    LLMs/finetune_class_HF_pytorch.py \
    --savedir /opt/data_dir/result/model \
    --configpath /opt/arxivDownload/rmme_config.toml \

