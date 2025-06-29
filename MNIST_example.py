import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

from client import Client
from server import Server
from fedavg import FederatedAvg


# -------------------------------
# Config class
# -------------------------------
class Config:
    def __init__(self):
        self.id = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD
        self.opt_args = {'lr': 0.01}
        self.local_epoch = 1
        self.local_batch_size = 32
        self.num_client = 10
        self.client_rate = 1.0  # All clients participate
        self.rounds = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Simple CNN Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------------
# Load & Partition MNIST
# -------------------------------
def load_and_partition_mnist(num_clients):
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split into equal chunks
    data_per_client = len(full_train_dataset) // num_clients
    client_datasets = []
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)

    for i in range(num_clients):
        subset_idx = indices[i * data_per_client:(i + 1) * data_per_client]
        client_datasets.append(Subset(full_train_dataset, subset_idx))

    return client_datasets

# -------------------------------
# Main
# -------------------------------
def main():
    config = Config()
    base_model = SimpleCNN().to(config.device)

    # Set up server
    server = Server(model=SimpleCNN())

    # Set up clients
    client_datasets = load_and_partition_mnist(config.num_client)
    clients = []

    for i in range(config.num_client):
        client_config = Config()
        client_config.id = i
        client = Client(config=client_config, model=SimpleCNN(), dataset=client_datasets[i], device=config.device)
        clients.append(client)

    # Federated Averaging Runner
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    fedavg = FederatedAvg(config=config,  server=server, clients=clients)
    fedavg.run(test_loader=test_loader)

if __name__ == "__main__":
    main()
