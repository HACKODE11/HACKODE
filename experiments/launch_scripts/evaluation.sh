#!/bin/bash
export WANDB_MODE=disabled
 

model=$1 
dir=$2
 
for data_offset in 0
do

    python ../evaluation.py \
        --config="../configs/${model}.py" \
        --config.result_prefix="${dir}" \
        --config.n_steps=500

done
