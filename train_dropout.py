import argparse
import csv
import itertools
import os
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from generate_colored_mnist import ColoredMNIST


class ResNetWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        base = torchvision.models.resnet18(weights = None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False )
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, 2)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc

        self.dropout = nn.Dropout(p = dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def get_model(dropout_rate = 0.0):
    return ResNetWithDropout(dropout_rate= dropout_rate)




def define_linear_probes():
    layer_configs = {
        'relu':   (64,  28),
        'layer1': (64,  28),
        'layer2': (128, 14),
        'layer3': (256,  7),
        'layer4': (512,  4),
        'avgpool': (512, 1),
    }

    color_probes = nn.ModuleDict()
    digit_probes = nn.ModuleDict()

    for name, (channels, spatial) in layer_configs.items():
        for probes in (color_probes, digit_probes):
            probes[name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2),
            )

    return color_probes, digit_probes

def apply_probes(color_probes, digit_probes, activation_dict, y_digit, y_color):
    probe_losses = []
    probe_correct = {}
    batch_size = y_digit.size(0)
    criterion = nn.CrossEntropyLoss()

    for name, probe in color_probes.items():
        act = activation_dict[name].detach()
        pred = probe(act)
        loss = criterion(pred, y_color)
        probe_losses.append(loss)
        correct = (pred.argmax(dim=1) == y_color).sum().item()
        probe_correct[f"{name}_color"] = correct

    for name, probe in digit_probes.items():
        act = activation_dict[name].detach()
        pred = probe(act)
        loss = criterion(pred, y_digit)
        probe_losses.append(loss)
        correct = (pred.argmax(dim=1) == y_digit).sum().item()
        probe_correct[f"{name}_digit"] = correct

    total_probe_loss = sum(probe_losses)
    return total_probe_loss, probe_correct, batch_size

def log_epoch_accuracy(epoch, epoch_correct, epoch_total, csv_path):
    row = {'epoch': epoch}
    for name, correct in epoch_correct.items():
        row[name] = correct / epoch_total

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main(args):
    timestamp = datetime.now().strftime('%m-%d-%H-%M')
    csv_path = f'{timestamp}_dropout{args.dropout_rate}_probe_accuracy.csv'

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = ColoredMNIST(root='~/datasets/mnist', split='train', color_correlation=0.9, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    
    model = get_model(dropout_rate = args.dropout_rate)

    activation_dict = {}
    def getActivation(name):
        def hook(model, input, output):
            activation_dict[name] = output.detach()
        return hook

    # register the forward hooks to get the activations
    h1 = model.relu.register_forward_hook(getActivation('relu'))
    h2 = model.layer1.register_forward_hook(getActivation('layer1'))
    h3 = model.layer2.register_forward_hook(getActivation('layer2'))
    h4 = model.layer3.register_forward_hook(getActivation('layer3'))
    h5 = model.layer4.register_forward_hook(getActivation('layer4'))
    h6 = model.avgpool.register_forward_hook(getActivation('avgpool'))

    color_probes, digit_probes = define_linear_probes()

    criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    probe_optimizer = optim.SGD(itertools.chain(color_probes.parameters(), digit_probes.parameters()), args.probe_lr)

    print("Beginning training...")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        epoch_correct = {}
        epoch_total = 0

        for X, y_digit, y_color in train_dataloader:
            model_optimizer.zero_grad()
            probe_optimizer.zero_grad()

            y_hat = model(X)
            loss = criterion(y_hat, y_digit)
            loss.backward()
            model_optimizer.step()
            train_loss += loss.item() * X.size(0)

            total_probe_loss, probe_correct, batch_size = apply_probes(color_probes, digit_probes, activation_dict, y_digit, y_color)
            total_probe_loss.backward()
            probe_optimizer.step()
            epoch_total += batch_size
            for name, correct in probe_correct.items():
                epoch_correct[name] = epoch_correct.get(name, 0) + correct

        log_epoch_accuracy(epoch, epoch_correct, epoch_total, csv_path)

        train_loss = train_loss / len(train_dataloader.dataset)
        print('Epoch: {} \tResnet Training Loss: {:.6f}'.format(epoch + 1, train_loss))

    print("Saving model...")
    torch.save(model.state_dict(), f"./dropout_{args.dropout_rate}_resnet.pt")

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    h5.remove()
    h6.remove()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--probe_lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    args = parser.parse_args()
    main(args)