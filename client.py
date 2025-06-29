import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Type, Tuple
from collections import OrderedDict

class Client:

    def __init__(self, config, model: nn.Module, dataset: Dataset, device = 'cpu'):

        self.id = config.id

        self.model = model
        self.local_dataset = dataset
        self.loss_fn : nn.modules.loss._Loss = config.loss_fn
        self.optimizer_class: Type[torch.optim.Optimizer] = config.optimizer
        self.optimizer = self.optimizer_class(self.model.parameters(), **config.opt_args)

        self.local_epoch = config.local_epoch
        self.local_batch_size = config.local_batch_size

        self.device = device
    
    def fetch_global_weights(self, global_weights):
        self.model.load_state_dict(global_weights)


    def update(self) -> Tuple[OrderedDict[str, torch.Tensor], int]:

        training_data = DataLoader(dataset= self.local_dataset, batch_size= self.local_batch_size, shuffle=True)
        self.model.to(self.device)

        for epoch in range(self.local_epoch):
            self.model.train()

            for batch_idx, batch in enumerate(training_data):
                
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict(), len(self.local_dataset)