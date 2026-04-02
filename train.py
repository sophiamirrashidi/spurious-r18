import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from generate_colored_mnist import ColoredMNIST

BATCH_SIZE = 32
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 40 

def train(args): 
    pass
    return 

def main(args):
    model = torchvision.models.resnet18(weights = None)

    train_dataset = ColoredMNIST(root='~/datasets/mnist', split='train', color_correlation=0.9)
    val_dataset   = ColoredMNIST(root='~/datasets/mnist', split='val',   color_correlation=0.7)
    test_dataset  = ColoredMNIST(root='~/datasets/mnist', split='test',  color_correlation=0.5)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    print("Beginning training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_dataloader:
            y_hat = model(X)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * X.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

    print("Saving model...")
    torch.save(model.state_dict(), "./task1.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="./data/dev", help="Path to the evaluation file (dev or test), defaults to dev")
    parser.add_argument("--test", action="store_true", help="Set this flag when evaluating on test data (no tags expected)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a trained .pt model file. If provided, skips training and runs prediction only.")
    args = parser.parse_args()
    main(args.eval_file, args.test, args.model_path)