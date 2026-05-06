import argparse
import csv
import itertools
import os
import torch
import pandas as pd
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from waterbirds import Waterbirds

def define_linear_probes():
    layer_configs = {
        'relu':   (64,  28),
        'layer1': (64,  28),
        'layer2': (128, 14),
        'layer3': (256,  7),
        'layer4': (512,  4),
        'avgpool': (512, 1),
    }

    water_probes = nn.ModuleDict()
    bird_probes = nn.ModuleDict()

    for name, (channels, spatial) in layer_configs.items():
        for probes in (water_probes, bird_probes):
            probes[name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2),
            )

    return water_probes, bird_probes

def apply_probes(water_probes, bird_probes, activation_dict, y_bird, y_water):
    probe_losses = []
    probe_correct = {}
    batch_size = y_bird.size(0)
    criterion = nn.CrossEntropyLoss()

    for name, probe in water_probes.items():
        act = activation_dict[name].detach()
        pred = probe(act)
        loss = criterion(pred, y_water)
        probe_losses.append(loss)
        correct = (pred.argmax(dim=1) == y_water).sum().item()
        probe_correct[f"{name}_water"] = correct

    for name, probe in bird_probes.items():
        act = activation_dict[name].detach()
        pred = probe(act)
        loss = criterion(pred, y_bird)
        probe_losses.append(loss)
        correct = (pred.argmax(dim=1) == y_bird).sum().item()
        probe_correct[f"{name}_bird"] = correct

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
    csv_path = f'{timestamp}_wb_probe_accuracy_{args.reg}_{args.weight_decay}.csv'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    df = pd.read_csv(f'{args.datapath}/metadata.csv')
    train_df = df[df['split'] == 0].reset_index(drop=True)
    train_dataset = Waterbirds(df=train_df, root=args.datapath, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)

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

    water_probes, bird_probes = define_linear_probes()

    criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    probe_optimizer = optim.SGD(itertools.chain(water_probes.parameters(), bird_probes.parameters()), args.probe_lr)

    print("Beginning training...")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        epoch_correct = {}
        epoch_total = 0

        for X, y_bird, y_water in train_dataloader:
            model_optimizer.zero_grad()
            probe_optimizer.zero_grad()

            y_hat = model(X)
            loss = criterion(y_hat, y_bird)
            loss.backward()
            model_optimizer.step()
            train_loss += loss.item() * X.size(0)

            total_probe_loss, probe_correct, batch_size = apply_probes(water_probes, bird_probes, activation_dict, y_bird, y_water)
            total_probe_loss.backward()
            probe_optimizer.step()
            epoch_total += batch_size
            for name, correct in probe_correct.items():
                epoch_correct[name] = epoch_correct.get(name, 0) + correct

        log_epoch_accuracy(epoch, epoch_correct, epoch_total, csv_path)

        train_loss = train_loss / len(train_dataloader.dataset)
        print('Epoch: {} \tResnet Training Loss: {:.6f}'.format(epoch + 1, train_loss))

    print("Saving model...")
    torch.save(model.state_dict(), "./resnet.pt")

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
    parser.add_argument("--reg", type=str, default=40)
    parser.add_argument("--datapath", type=str, default="../waterbird_complete95_forest2water2")
    args = parser.parse_args()
    main(args)
