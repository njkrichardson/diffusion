from typing import Optional, Sequence, Tuple 

import torch 
import torch.nn.functional as F 

from constants import Tensor 

def cosine_schedule(timesteps: Tensor, s: float) -> Tensor: 
    steps: Tensor = timesteps + 1 
    x: Tensor = torch.linspace(0, timesteps, steps) 
    cumulative_product: Tensor = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) **2 
    cumulative_product = cumulative_product / cumulative_product[0]
    betas: Tensor = 1 - (cumulative_product[1:] / cumulative_product[:-1])
    return torch.clip(betas, 1e-04, 0.9999) 

def linear_schedule(timesteps: Tensor, **kwargs) -> Tensor: 
    start: float = kwargs.get("start", 1e-04)
    end: float = kwargs.get("end", 2e-02)
    return torch.linspace(start, end, timesteps) 

def extract_timestep(batch: Tensor, timesteps: Tensor, shape: Tuple[int, ...]) -> Tensor: 
    batch_size: int = timesteps.shape[0]
    out: Tensor = batch.gather(-1, timesteps.cpu())
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timesteps.device)

def bind_diffusion(num_timesteps: int, schedule: callable, **kwargs) -> Tuple[callable, callable]: 
    betas: Tensor = schedule(num_timesteps, **kwargs) 
    alphas: Tensor = 1. - betas 
    alphas_cumulative_product: Tensor = torch.cumprod(alphas, axis=0)
    alphas_cumulative_product_previous: Tensor = F.pad(alphas_cumulative_product[:-1], (1, 0), value=1.0)
    sqrt_reciprocal_alphas: Tensor = torch.sqrt(1.0 / alphas) 

    sqrt_alphas_cumulative_product: Tensor = torch.sqrt(alphas_cumulative_product)
    sqrt_one_minus_alphas_cumulative_product: Tensor = torch.sqrt(1. - alphas_cumulative_product)

    posterior_variance: Tensor = betas * (1. - alphas_cumulative_product_previous) / (1. - alphas_cumulative_product)

    def diffuse(x: Tensor, timesteps: Tensor, noise: Optional[Tensor]=None) -> Tensor: 
        if noise is None: 
            noise: Tensor = torch.randn_like(x) 

        sqrt_alphas_cumulative_product_t: Tensor = extract_timestep(sqrt_alphas_cumulative_product, timesteps, x.shape)
        sqrt_one_minus_alphas_cumulative_product_t: Tensor = extract_timestep(sqrt_one_minus_alphas_cumulative_product, timesteps, x.shape)

        return sqrt_alphas_cumulative_product_t * x + sqrt_one_minus_alphas_cumulative_product_t * noise 

    @torch.no_grad()
    def sample(decoder: callable, x: Tensor, timesteps: Tensor, t_index: int) -> Tensor: 
        betas_t: Tensor = extract_timestep(betas, timesteps, x.shape)
        sqrt_one_minus_alphas_cumulative_product_t: Tensor = extract_timestep(sqrt_one_minus_alphas_cumulative_product, timesteps, x.shape)
        sqrt_reciprocal_alphas_t: Tensor = extract_timestep(sqrt_reciprocal_alphas, timesteps, x.shape)
        
        predicted_noise: Tensor = decoder(x, timesteps)
        mean: Tensor = sqrt_reciprocal_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumulative_product_t)

        if t_index == 0:
            return mean
        else:
            posterior_variance_t: Tensor = extract_timestep(posterior_variance, timesteps, x.shape)
            noise: Tensor = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample_batched(decoder: callable, shape: Sequence[int]) -> Tensor:
        device: torch.device = next(decoder.parameters()).device

        batch_size: int = shape[0]

        # start from pure noise (for each example in the batch)
        z: Tensor = torch.randn(shape, device=device)
        samples: List[Tensor] = []

        for i in reversed(range(0, num_timesteps)):
            z = sample(decoder, z, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
            samples.append(z.cpu().numpy())
        return samples

    return diffuse, sample_batched
