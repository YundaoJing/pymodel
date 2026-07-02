import pyexp
import numpy
import torch
import torch.nn
from .basemodel import BaseModel

class NN(BaseModel):

    def __init__(self, dict_dataset: dict, device='cpu'):
        super().__init__(
            dict_dataset=dict_dataset,
            device=device)
        # initial layer
        self.modulelist = torch.nn.ModuleList([
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),])
        self.to(self.device)

    def forward(self, tensor_input: torch.Tensor):

        tensor_iter = tensor_input
        for layer in self.modulelist:
            tensor_iter = layer(tensor_iter)
        tensor_output = tensor_iter

        return tensor_output