import torch.nn as nn


# Layer spatial dimensions per dataset
LAYER_CONFIGS = {
    'cmnist': {
        'relu':    (64, 28),
        'layer1':  (64, 28),
        'layer2':  (128, 14),
        'layer3':  (256, 7),
        'layer4':  (512, 4),
        'avgpool': (512, 1),
    },
    'waterbirds': {
        'relu':    (64, 56),
        'layer1':  (64, 56),
        'layer2':  (128, 28),
        'layer3':  (256, 14),
        'layer4':  (512, 7),
        'avgpool': (512, 1),
    },
}

# Probe attribute names per dataset
PROBE_NAMES = {
    'cmnist': ('color', 'digit'),
    'waterbirds': ('water', 'bird'),
}


def define_linear_probes(dataset):
    """
    Create two sets of linear probes for the given dataset.

    Returns (spurious_probes, core_probes) — e.g. (color_probes, digit_probes)
    or (water_probes, bird_probes).
    """
    layer_configs = LAYER_CONFIGS[dataset]

    spurious_probes = nn.ModuleDict()
    core_probes = nn.ModuleDict()

    for name, (channels, spatial) in layer_configs.items():
        for probes in (spurious_probes, core_probes):
            probes[name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2),
            )

    return spurious_probes, core_probes


def apply_probes(spurious_probes, core_probes, activation_dict, y_core, y_spurious, dataset):
    """
    Compute probe losses and accuracies.

    Args:
        spurious_probes: probes for the spurious attribute (color/water)
        core_probes: probes for the core attribute (digit/bird)
        activation_dict: layer name -> activation tensor
        y_core: labels for core task (digit/bird)
        y_spurious: labels for spurious attribute (color/water)
        dataset: 'cmnist' or 'waterbirds'

    Returns:
        total_probe_loss, probe_correct dict, batch_size
    """
    spurious_name, core_name = PROBE_NAMES[dataset]
    probe_losses = []
    probe_correct = {}
    batch_size = y_core.size(0)
    criterion = nn.CrossEntropyLoss()

    for name, probe in spurious_probes.items():
        act = activation_dict[name].detach()
        pred = probe(act)
        loss = criterion(pred, y_spurious)
        probe_losses.append(loss)
        correct = (pred.argmax(dim=1) == y_spurious).sum().item()
        probe_correct[f"{name}_{spurious_name}"] = correct

    for name, probe in core_probes.items():
        act = activation_dict[name].detach()
        pred = probe(act)
        loss = criterion(pred, y_core)
        probe_losses.append(loss)
        correct = (pred.argmax(dim=1) == y_core).sum().item()
        probe_correct[f"{name}_{core_name}"] = correct

    total_probe_loss = sum(probe_losses)
    return total_probe_loss, probe_correct, batch_size
