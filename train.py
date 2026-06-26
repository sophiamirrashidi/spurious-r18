"""
Unified training script for spurious correlation experiments.

Supports both datasets (Colored MNIST, Waterbirds) and all regularization
strategies (L1, L2/weight decay, Dropout, or none) via CLI flags.

Examples:
    python train.py --dataset cmnist --reg l2 --weight_decay 1e-3
    python train.py --dataset waterbirds --reg l1 --l1_lambda 1e-3
    python train.py --dataset cmnist --reg dropout --dropout_rate 0.3
    python train.py --dataset waterbirds --reg none
"""

import argparse
import itertools
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.models import get_resnet18_cmnist, get_resnet18_waterbirds
from src.probes import define_linear_probes, apply_probes
from src.regularization import l1_regularization
from src.logging_utils import log_epoch_accuracy
from src.datasets import ColoredMNIST, Waterbirds


def load_cmnist(args):
    """Load Colored MNIST training data."""
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_dataset = ColoredMNIST(
        root=args.data_root,
        split='train',
        color_correlation=0.9,
        transform=transform,
    )
    return train_dataset


def load_waterbirds(args):
    """Load Waterbirds training data."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    df = pd.read_csv(os.path.join(args.datapath, 'metadata.csv'))
    train_df = df[df['split'] == 0].reset_index(drop=True)
    train_dataset = Waterbirds(df=train_df, root=args.datapath, transform=transform)
    return train_dataset


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Output path ---
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    if args.reg == 'none':
        reg_label = 'baseline'
    elif args.reg == 'l1':
        reg_label = f'l1_{args.l1_lambda}'
    elif args.reg == 'l2':
        reg_label = f'l2_{args.weight_decay}'
    elif args.reg == 'dropout':
        reg_label = f'dropout_{args.dropout_rate}'

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'{timestamp}_probe_accuracy_{reg_label}.csv')

    # --- Data ---
    if args.dataset == 'cmnist':
        train_dataset = load_cmnist(args)
    else:
        train_dataset = load_waterbirds(args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    dropout_rate = args.dropout_rate if args.reg == 'dropout' else 0.0
    if args.dataset == 'cmnist':
        model = get_resnet18_cmnist(dropout_rate=dropout_rate)
    else:
        model = get_resnet18_waterbirds(dropout_rate=dropout_rate)
    model = model.to(device)

    # --- Hooks ---
    activation_dict = {}

    def get_activation(name):
        def hook(module, input, output):
            activation_dict[name] = output
        return hook

    hooks = []
    for layer_name in ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
        layer = getattr(model, layer_name)
        hooks.append(layer.register_forward_hook(get_activation(layer_name)))

    # --- Probes ---
    spurious_probes, core_probes = define_linear_probes(args.dataset)
    spurious_probes = spurious_probes.to(device)
    core_probes = core_probes.to(device)

    # --- Optimizers ---
    weight_decay = args.weight_decay if args.reg == 'l2' else 0.0
    model_optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=weight_decay,
    )
    probe_optimizer = optim.SGD(
        itertools.chain(spurious_probes.parameters(), core_probes.parameters()),
        lr=args.probe_lr,
    )

    # --- Training loop ---
    criterion = nn.CrossEntropyLoss()

    print(f"Beginning training: dataset={args.dataset}, reg={args.reg}, epochs={args.num_epochs}")
    for epoch in range(args.num_epochs):
        model.train()
        spurious_probes.train()
        core_probes.train()

        train_loss = 0.0
        epoch_correct = {}
        epoch_total = 0

        for batch in train_dataloader:
            X, y_core, y_spurious = batch
            X = X.to(device, non_blocking=True)
            y_core = y_core.to(device, non_blocking=True)
            y_spurious = y_spurious.to(device, non_blocking=True)

            model_optimizer.zero_grad()
            probe_optimizer.zero_grad()

            y_hat = model(X)
            loss = criterion(y_hat, y_core)

            # Apply L1 regularization if selected
            if args.reg == 'l1' and args.l1_lambda > 0:
                loss = loss + args.l1_lambda * l1_regularization(model)

            loss.backward()
            model_optimizer.step()
            train_loss += loss.item() * X.size(0)

            # Train probes
            total_probe_loss, probe_correct, batch_size = apply_probes(
                spurious_probes, core_probes, activation_dict,
                y_core, y_spurious, args.dataset,
            )
            total_probe_loss.backward()
            probe_optimizer.step()

            epoch_total += batch_size
            for name, correct in probe_correct.items():
                epoch_correct[name] = epoch_correct.get(name, 0) + correct

        log_epoch_accuracy(epoch, epoch_correct, epoch_total, csv_path)

        train_loss = train_loss / len(train_dataloader.dataset)
        print(f'Epoch: {epoch + 1}\tTraining Loss: {train_loss:.6f}')

    # --- Save model ---
    save_path = os.path.join(args.output_dir, f'resnet_{reg_label}.pt')
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)

    # --- Cleanup hooks ---
    for h in hooks:
        h.remove()

    print(f"Done. Probe accuracies saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 with linear probes")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, choices=['cmnist', 'waterbirds'],
                        help="Dataset to train on")
    parser.add_argument("--data_root", type=str, default="~/datasets/mnist",
                        help="Root directory for MNIST data (cmnist only)")
    parser.add_argument("--datapath", type=str, default="../waterbird_complete95_forest2water2",
                        help="Path to Waterbirds dataset (waterbirds only)")

    # Regularization
    parser.add_argument("--reg", type=str, default="none", choices=['none', 'l1', 'l2', 'dropout'],
                        help="Regularization strategy")
    parser.add_argument("--l1_lambda", type=float, default=1e-5,
                        help="L1 regularization weight")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="L2 weight decay for SGD optimizer")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout probability between residual blocks")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01, help="Model learning rate")
    parser.add_argument("--probe_lr", type=float, default=0.01, help="Probe learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_epochs", type=int, default=40)

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for outputs (default: results/<dataset>/<reg>)")

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        reg_dir = args.reg if args.reg != 'none' else 'baseline'
        args.output_dir = os.path.join('results', args.dataset, reg_dir)

    main(args)
