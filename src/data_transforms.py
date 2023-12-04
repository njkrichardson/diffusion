from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from constants import Module, Tensor
from utils import deserialize

def close_transform(image_dimension: int) -> Tuple[Module]: 
    transform: Module = Compose([
        Resize(image_dimension),
        CenterCrop(image_dimension),
        Lambda(lambda value: (value * 2) - 1),
    ])
    reverse_transform: Module = Compose([
         Lambda(lambda t: (t + 1) / 2),
         Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
         Lambda(lambda t: t * 255.),
         Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])
    return transform, reverse_transform

class MonochromeDataset(Dataset): 
    def __init__(self, num_observations: int, observation_dimension: int, device: torch.device, transform: Optional[callable]=None): 
        data_distribution: callable = lambda shape: torch.arange(2)[torch.multinomial(torch.tensor([0.2, 0.8]), torch.prod(torch.tensor(shape)), replacement=True)].reshape(*shape)
        observations: Tensor = data_distribution((num_observations, observation_dimension, observation_dimension)).to(device)
        self.num_observations: int = num_observations
        self.observations = observations.type(torch.float) 
        self.transform = transform 

    def __len__(self) -> int: 
        return self.num_observations

    def __getitem__(self, index: Union[Tensor, int]) -> Tensor: 
        if torch.is_tensor(index): 
            index.tolist()
            x: Tensor = self.observations[index] 
        else: 
            x: Tensor = self.observations[index][None, ...]

        if self.transform: 
            x = self.transform(x) 

        return x 

class ControlDataset(Dataset): 
    def __init__(self, path: Path, transform: Module, device: torch.device): 
        self.path: Path = path 
        self.device: torch.device = device 
        self.transform: Module = transform 
        self.data: Tensor = torch.from_numpy(deserialize(self.path))
        
        # TODO use FP32 on generation 

    def __len__(self) -> int: 
        return self.data.shape[0] 

    @property 
    def frame_width(self) -> int: 
        return self.data[0].shape[3] 

    @property 
    def frame_height(self) -> int: 
        return self.data[0].shape[2] 

    @property 
    def frames_per_video(self) -> int: 
        return self.data[0].shape[1] 

    @property 
    def num_channels(self) -> int: 
        return self.data[0].shape[0] 

    def __getitem__(self, index: Union[Tensor, int]) -> Tensor: 
        if torch.is_tensor(index): 
            index.tolist()
            x: Tensor = self.data[index] 
        else: 
            x: Tensor = self.data[index][None, ...]

        if self.transform: 
            x = x.type(torch.float32) 
            x = self.transform(x) 

        return x.to(self.device) 
