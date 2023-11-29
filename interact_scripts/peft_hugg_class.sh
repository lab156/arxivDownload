#!/bin/bash

# To be run in interact mode potentially with multiple gpus

singularity run --nv \
    --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir \
    $PROJECT/singul/runnerTransHFPerf.sif \
    python3 embed/peft_tuning_and_lora_seq_cls.py \
    --savedir /opt/data_dir/result/model \
    --configpath /opt/arxivDownload/config_version_control.toml
