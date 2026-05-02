import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from generate_colored_mnist import ColoredMNIST


def get_model():
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 2)
    return model


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    correct = 0

    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for X, y_digit, y_color in dataloader:
            X = X.to(device)
            y_digit = y_digit.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            total += y_digit.size(0)
            correct += (preds == y_digit).sum().item()

            tp += ((preds == 1) & (y_digit == 1)).sum().item()
            fp += ((preds == 1) & (y_digit == 0)).sum().item()
            fn += ((preds == 0) & (y_digit == 1)).sum().item()

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return accuracy, precision, recall


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    test_dataset = ColoredMNIST(
        root='~/datasets/mnist',
        split='test',
        color_correlation=args.color_correlation,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model()
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

    model.to(device)

    accuracy, precision, recall = evaluate(model, test_loader, device)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--color_correlation", type=float, default=0.5)
  
    args = parser.parse_args()
    main(args)
