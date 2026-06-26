#!/bin/bash
# Sweep over regularization strategies for Colored MNIST

set -e

echo "=== Colored MNIST: L2 Regularization ==="
for wd in 1e-5 1e-3 0.1; do
    echo "  weight_decay=$wd"
    python train.py --dataset cmnist --reg l2 --weight_decay $wd --num_epochs 40
done

echo "=== Colored MNIST: L1 Regularization ==="
for l1 in 1e-5 1e-3 1e-2; do
    echo "  l1_lambda=$l1"
    python train.py --dataset cmnist --reg l1 --l1_lambda $l1 --num_epochs 40
done

echo "=== Colored MNIST: Dropout ==="
for dr in 0.1 0.3 0.5; do
    echo "  dropout_rate=$dr"
    python train.py --dataset cmnist --reg dropout --dropout_rate $dr --num_epochs 40
done

echo "Done! Results saved to results/colored_mnist/"
