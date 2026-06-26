#!/bin/bash
# Sweep over regularization strategies for Waterbirds

set -e

DATAPATH="${WATERBIRDS_PATH:-../waterbird_complete95_forest2water2}"

echo "=== Waterbirds: Baseline (no regularization) ==="
python train.py --dataset waterbirds --reg none --datapath "$DATAPATH" --num_epochs 40

echo "=== Waterbirds: L2 Regularization ==="
for wd in 1e-5 1e-3 0.1; do
    echo "  weight_decay=$wd"
    python train.py --dataset waterbirds --reg l2 --weight_decay $wd --datapath "$DATAPATH" --num_epochs 40
done

echo "=== Waterbirds: L1 Regularization ==="
for l1 in 1e-5 1e-3 1e-2; do
    echo "  l1_lambda=$l1"
    python train.py --dataset waterbirds --reg l1 --l1_lambda $l1 --datapath "$DATAPATH" --num_epochs 40
done

echo "=== Waterbirds: Dropout ==="
for dr in 0.1 0.3 0.5; do
    echo "  dropout_rate=$dr"
    python train.py --dataset waterbirds --reg dropout --dropout_rate $dr --datapath "$DATAPATH" --num_epochs 10
done

echo "Done! Results saved to results/waterbirds/"
