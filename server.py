import torch
import torch.nn as nn
from typing import List, Tuple
from collections import OrderedDict
import copy

class Server:

    def __init__(self, model: nn.Module):
        self.global_model = model
        self.global_model_weights = copy.deepcopy(model.state_dict())
        

    def update_global(self, client_updates: List[Tuple[OrderedDict[str, torch.Tensor], int]]):
        total_samples = sum(num_samples for _, num_samples in client_updates)

        new_weights = {k: torch.zeros_like(v) for k, v in self.global_model_weights.items()}

        for client_weights, num_samples in client_updates:
            weight_factor = num_samples / total_samples
            for k in new_weights:
                new_weights[k] += client_weights[k].to('cpu') * weight_factor

        self.global_model.load_state_dict(new_weights)
        self.global_model_weights = copy.deepcopy(new_weights)

    def get_global_model_weights(self) -> OrderedDict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model_weights)
