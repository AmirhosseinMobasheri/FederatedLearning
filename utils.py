import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_model(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str = 'cpu') -> float:
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate_model_accuracy(model: nn.Module, dataloader: DataLoader, device: str = 'cpu') -> float:
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0
