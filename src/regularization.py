import torch


def l1_regularization(model):
    """Compute L1 norm of all non-bias weight parameters."""
    l1_norm = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            l1_norm += param.abs().sum()
    return l1_norm
