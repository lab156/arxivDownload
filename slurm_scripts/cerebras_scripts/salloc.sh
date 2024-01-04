#! /bin/bash

srun --pty --cpus-per-task=28 --kill-on-bad-exit \
    singularity shell --cleanenv --bind \
    /local1/cerebras/data,/local2/cerebras/data,/local3/cerebras/data,/local4/cerebras/data,$PROJECT \
    /ocean/neocortex/cerebras/cbcore_latest.sif
