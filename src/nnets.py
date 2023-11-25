from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch 
import torch.nn as nn 

from constants import Tensor

Module: type = nn.Module 

@dataclass 
class AffineNetConfig: 
    input_dimension: Optional[int]=1 
    hidden_dimensions: Optional[Sequence[int]]=(100, 100) 

class AffineNet(Module): 
    def __init__(self, config: AffineNetConfig): 
        super().__init__()
        self.config: AffineNetConfig = config 
        # TODO, handle 2D transforms
        self.config.input_dimension = torch.prod(torch.tensor(self.config.input_dimension))
        layers: Sequence[Module] = [
                nn.Linear(self.config.input_dimension + 1, self.config.hidden_dimensions[0], bias=True), 
                ]
        for input_dimension, output_dimension in zip(self.config.hidden_dimensions[:-1], self.config.hidden_dimensions[1:]): 
            layers.append(nn.Linear(input_dimension, output_dimension)) 
            layers.append(nn.LeakyReLU())

        if self.config.hidden_dimensions[-1] != self.config.input_dimension: 
            layers.append(nn.Linear(self.config.hidden_dimensions[-1], self.config.input_dimension))

        self.net: Module = nn.Sequential(*layers) 

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor: 
        x_shape: Sequence[int] = x.shape
        x = x.reshape(1, -1)
        input: Tensor = torch.cat((x, timestep.reshape(1, 1)), -1) 
        return self.net(input).reshape(x_shape)

@dataclass 
class MLPConfig: 
    input_dimension: Optional[int]=1 
    hidden_dimensions: Optional[Sequence[int]]=(4, 4, 4) 
    activation: Optional[Module]=nn.LeakyReLU
    activate_last: Optional[bool]=False

class MLP(Module): 
    def __init__(self, config: MLPConfig): 
        super().__init__()
        self.config: MLPConfig = config 
        self.activation: Module = self.config.activation
        layers: Sequence[Module] = [
                nn.Linear(self.config.input_dimension + 1, self.config.hidden_dimensions[0], bias=True), 
                self.activation()
                ]
        for input_dimension, output_dimension in zip(self.config.hidden_dimensions[:-1], self.config.hidden_dimensions[1:]): 
            layers.append(nn.Linear(input_dimension, output_dimension)) 
            layers.append(self.activation())

        if self.config.hidden_dimensions[-1] != self.config.input_dimension: 
            layers.append(nn.Linear(self.config.hidden_dimensions[-1], self.config.input_dimension))

        if activate_last: 
            layers.append(self.activation())

        self.net: Module = nn.Sequential(*layers) 

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor: 
        input: Tensor = torch.cat((x, timestep)) 
        return self.net(input) 
