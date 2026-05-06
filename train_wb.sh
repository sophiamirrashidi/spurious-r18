#!/bin/bash

python train_waterbirds.py \
    --batch_size 32 \
    --lr 0.01 \
    --probe_lr 0.01 \
    --momentum 0.9 \
    --weight_decay 0.1\
    --num_epochs 10 \
    --reg "l2"

