#!/bin/bash

python train.py \
    --batch_size 32 \
    --lr 0.01 \
    --probe_lr 0.01 \
    --momentum 0.9 \
    --weight_decay 1e-5 \
    --num_epochs 40
