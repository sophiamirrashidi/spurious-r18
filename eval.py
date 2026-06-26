"""
Evaluate a trained model on Colored MNIST or Waterbirds test set.

Examples:
    python eval.py --dataset cmnist --model_path results/colored_mnist/l2/resnet_l2_1e-3.pt
    python eval.py --dataset waterbirds --model_path results/waterbirds/l1/resnet_l1_1e-5.pt
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.models import get_resnet18_cmnist, get_resnet18_waterbirds
from src.datasets import ColoredMNIST, Waterbirds


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y_core, y_spurious = batch
            X = X.to(device)
            y_core = y_core.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            total += y_core.size(0)
            correct += (preds == y_core).sum().item()

            tp += ((preds == 1) & (y_core == 1)).sum().item()
            fp += ((preds == 1) & (y_core == 0)).sum().item()
            fn += ((preds == 0) & (y_core == 1)).sum().item()

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, precision, recall


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cmnist':
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        test_dataset = ColoredMNIST(
            root=args.data_root,
            split='test',
            color_correlation=args.color_correlation,
            transform=transform,
        )
        model = get_resnet18_cmnist(dropout_rate=args.dropout_rate)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        df = pd.read_csv(os.path.join(args.datapath, 'metadata.csv'))
        test_df = df[df['split'] == 2].reset_index(drop=True)
        test_dataset = Waterbirds(df=test_df, root=args.datapath, transform=transform)
        model = get_resnet18_waterbirds(dropout_rate=args.dropout_rate)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.to(device)

    accuracy, precision, recall = evaluate(model, test_loader, device)

    print(f"Dataset:   {args.dataset}")
    print(f"Model:     {args.model_path}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--dataset", type=str, required=True, choices=['cmnist', 'waterbirds'])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.0,
                        help="Set this to match the dropout used during training")
    # cmnist options
    parser.add_argument("--color_correlation", type=float, default=0.5)
    parser.add_argument("--data_root", type=str, default="~/datasets/mnist")
    # waterbirds options
    parser.add_argument("--datapath", type=str, default="../waterbird_complete95_forest2water2")

    args = parser.parse_args()
    main(args)
