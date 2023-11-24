import torch 
import torch.nn.functional as F 

from constants import Tensor

def score_matching_objective(decoder: callable, encoder: callable, x: Tensor, timesteps: Tensor, noise: Optional[Tensor]=None) -> Tensor: 
    if noise is None: 
        noise: Tensor = torch.randn_like(x) 

    z: Tensor = encoder(x, timesteps, noise=noise) 
    predicted_noise: Tensor = decoder(z, timesteps) 

    return F.l1_loss(noise, predicted_noise) 
