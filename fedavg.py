import torch.nn as nn
from server import Server
from client import Client
from typing import List
import random
from torch.utils.data import DataLoader
from utils import evaluate_model, evaluate_model_accuracy


class FederatedAvg:


    def __init__(self, config, server: Server, clients: List[Client]):
        
        self.num_client = config.num_client
        self.client_rate = config.client_rate

        self.rounds = config.rounds

        self.server = server
        self.clients = clients


    def run(self, test_loader=None):


        for round in range(self.rounds):

            num_selected_clients = int(self.client_rate * self.num_client)
            selected_clients = random.sample(self.clients, num_selected_clients)
            
            selected_client_updates = []
            global_weights = self.server.get_global_model_weights()

            for selected_client in selected_clients:

                selected_client.fetch_global_weights(global_weights)

                W, n = selected_client.update()
                selected_client_updates.append((W, n))
            
            self.server.update_global(selected_client_updates)
            print(f"Round {round + 1}/{self.rounds} complete. {len(selected_clients)} clients participated.")

            if test_loader is not None:
                val_loss = evaluate_model(
                    model=self.server.global_model,
                    dataloader=test_loader,
                    loss_fn=self.clients[0].loss_fn,
                    device=self.clients[0].device
                )
                print(f"Validation loss after round {round + 1}: {val_loss:.4f}")

                val_acc = evaluate_model_accuracy(
                    model=self.server.global_model,
                    dataloader=test_loader,
                    device=self.clients[0].device
                )
                print(f"Validation accuracy after round {round + 1}: {val_acc:.4f}")




