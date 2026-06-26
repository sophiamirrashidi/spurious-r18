import torch
import torch.nn as nn
import torchvision


class ResNetWithDropout(nn.Module):
    """ResNet18 with dropout inserted between each residual block."""

    def __init__(self, base_model, dropout_rate=0.0):
        super().__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

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


def get_resnet18_cmnist(dropout_rate=0.0):
    """ResNet18 for 28x28 Colored MNIST (3x3 conv1, no maxpool)."""
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 2)

    if dropout_rate > 0:
        model = ResNetWithDropout(model, dropout_rate=dropout_rate)
    return model


def get_resnet18_waterbirds(dropout_rate=0.0):
    """Standard ResNet18 for 224x224 Waterbirds images."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(512, 2)

    if dropout_rate > 0:
        model = ResNetWithDropout(model, dropout_rate=dropout_rate)
    return model
