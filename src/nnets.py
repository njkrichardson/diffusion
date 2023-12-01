from dataclasses import dataclass
from functools import partial 
from inspect import isfunction 
import math 
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import einsum 

from constants import Tensor

Module: type = nn.Module 

def exists(x):
    return x is not None

exists: Callable[[Any], bool] = lambda x: x is not None

def default(value: Any, default: Any) -> Any:
    if exists(value):
        return value
    return default() if isfunction(default) else default

def num_to_groups(num: int, divisor: int) -> List[int]:
    num_groups: int = num // divisor
    remainder = num % divisor
    groups: List[int] = [divisor] * num_groups
    if remainder > 0:
        groups.append(remainder)
    return group

class Residual(Module):
    def __init__(self, fn: callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def upsample(input_dimension: int, output_dimension: Optional[int]=None, scale_factor: Optional[int]=2, **kwargs) -> Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode="nearest"),
        nn.Conv2d(input_dimension, default(output_dimension, input_dimension), kwargs.get("kernel_size", 3), padding=kwargs.get("padding", 1)),
    )

def downsample(input_dimension: int, output_dimension: Optional[int]=None, **kwargs) -> Module:
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(input_dimension * 4, default(output_dimension, input_dimension), kwargs.get("kernel_size", 1)),
    )

class SinusoidalPositionEmbeddings(Module):
    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension

    def forward(self, time: int) -> Tensor:
        device: torch.device = time.device
        half_dim: int = self.embedding_dimension // 2
        embeddings: Tensor = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    """https://arxiv.org/abs/1903.10520
    """
    def forward(self, x: Tensor) -> Tensor:
        eps: float = 1e-05 if x.dtype == torch.float32 else 1e-03

        weight: Tensor = self.weight
        mean: Tensor = reduce(weight, "o ... -> o 1 1 1", "mean")
        variance: Tensor = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight: Tensor = (weight - mean) / (variance + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(Module):
    def __init__(self, input_dimension: int, output_dimension: int, num_groups: Optional[int]=8, **kwargs):
        super().__init__()
        self.projection: Module = WeightStandardizedConv2d(input_dimension, output_dimension, kwargs.get("kernel_size", 3), padding=kwargs.get("padding", 1))
        self.normalization: Module = nn.GroupNorm(num_groups, output_dimension)
        self.activation: Module = nn.SiLU()

    def forward(self, x: Tensor, scale_shift: Optional[Tuple[Tensor]]=None) -> Tensor:
        x = self.projection(x)
        x = self.normalization(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x


class ResnetBlock(Module):
    """https://arxiv.org/abs/1512.03385
    """
    def __init__(self, input_dimension: int, output_dimension: int, *, temporal_embedding_dimension: Optional[int]=None, num_groups: Optional[int]=8):
        super().__init__()
        self.mlp: Module = (
            nn.Sequential(nn.SiLU(), nn.Linear(temporal_embedding_dimension, output_dimension * 2))
            if exists(temporal_embedding_dimension)
            else None
        )
        self.block1: Module = Block(input_dimension, output_dimension, num_groups=num_groups)
        self.block2: Module = Block(output_dimension, output_dimension, num_groups=num_groups)
        self.residual_conv: Module = nn.Conv2d(input_dimension, output_dimension, 1) if input_dimension != output_dimension else nn.Identity()

    def forward(self, x: Tensor, temporal_embedding: Optional[Tensor]=None) -> Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(temporal_embedding):
            temporal_embedding = self.mlp(temporal_embedding) 
            temporal_embedding = rearrange(temporal_embedding, "b c -> b c 1 1")
            scale_shift = temporal_embedding.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.residual_conv(x)

class Attention(Module):
    def __init__(self, input_dimension: int, num_heads: Optional[int]=4, head_dimension: Optional[int]=32):
        super().__init__()
        self.scale = head_dimension**-0.5
        self.num_heads = num_heads
        hidden_dimension: int = head_dimension * num_heads

        self.to_qkv: Module = nn.Conv2d(input_dimension, hidden_dimension * 3, 1, bias=False)
        self.to_out: Module = nn.Conv2d(hidden_dimension, input_dimension, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_channels, height, width = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=height, y=width)
        return self.to_out(out)

class LinearAttention(Module):
    def __init__(self, input_dimension: int, num_heads: Optional[int]=4, head_dimension: Optional[int]=32):
        super().__init__()
        self.scale = head_dimension**-0.5
        self.num_heads = num_heads
        hidden_dimension: int = head_dimension * num_heads
        
        self.to_qkv: Module = nn.Conv2d(input_dimension, hidden_dimension * 3, 1, bias=False)
        self.to_out: Module = nn.Sequential(nn.Conv2d(hidden_dimension, input_dimension, 1), nn.GroupNorm(1, input_dimension))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_channels, height, width = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.num_heads, x=height, y=width)
        return self.to_out(out)

class PreNorm(Module):
    def __init__(self, input_dimension: int, fn: callable):
        super().__init__()
        self.fn = fn
        self.normalization: Module = nn.GroupNorm(1, input_dimension)

    def forward(self, x: Tensor) -> Tensor:
        x = self.normalization(x)
        return self.fn(x)

class UNet(Module):
    def __init__(
        self,
        input_dimension: int,
        init_dimension: Optional[int]=None,
        output_dimension: Optional[int]=None,
        scale_factors: Tuple[int]=(1, 2, 4, 8),
        num_channels: Optional[int]=1,
        self_condition: Optional[bool]=False,
        resnet_block_groups: Optional[int]=4,
        **kwargs, 
    ):
        super().__init__()

        # determine dimensions
        self.num_channels = num_channels
        self.self_condition = self_condition
        input_channels = num_channels * (2 if self_condition else 1)

        init_dimension = default(init_dimension, input_dimension)
        self.init_conv: Module = nn.Conv2d(input_channels, init_dimension, kwargs.get("kernel_size", 1), padding=kwargs.get("padding", 0)) 

        dimensions: List[int] = [init_dimension, *map(lambda m: input_dimension * m, scale_factors)]
        input_output_dimensions: List[Tuple[int]] = list(zip(dimensions[:-1], dimensions[1:]))

        block_klass: Module = partial(ResnetBlock, num_groups=resnet_block_groups)

        # time embeddings
        temporal_embedding_dimension: int = input_dimension * 4

        self.time_mlp: Module = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dimension),
            nn.Linear(input_dimension, temporal_embedding_dimension),
            nn.GELU(),
            nn.Linear(temporal_embedding_dimension, temporal_embedding_dimension),
        )

        # layers
        self.downsamplers: nn.ModuleList = nn.ModuleList([])
        self.upsamplers: nn.ModuleList = nn.ModuleList([])
        num_transformations: int = len(input_output_dimensions)

        for ind, (in_dimension, out_dimension) in enumerate(input_output_dimensions):
            is_last: bool = ind >= (num_transformations - 1)

            self.downsamplers.append(
                nn.ModuleList(
                    [
                        block_klass(in_dimension, in_dimension, temporal_embedding_dimension=temporal_embedding_dimension),
                        block_klass(in_dimension, in_dimension, temporal_embedding_dimension=temporal_embedding_dimension),
                        Residual(PreNorm(in_dimension, LinearAttention(in_dimension))),
                        downsample(in_dimension, out_dimension)
                        if not is_last
                        else nn.Conv2d(in_dimension, out_dimension, 3, padding=1),
                    ]
                )
            )

        bottleneck_dimension: int = dimensions[-1]
        self.mid_block1 = block_klass(bottleneck_dimension, bottleneck_dimension, temporal_embedding_dimension=temporal_embedding_dimension)
        self.mid_attn = Residual(PreNorm(bottleneck_dimension, Attention(bottleneck_dimension)))
        self.mid_block2 = block_klass(bottleneck_dimension, bottleneck_dimension, temporal_embedding_dimension=temporal_embedding_dimension)

        for ind, (in_dimension, out_dimension) in enumerate(reversed(input_output_dimensions)):
            is_last: bool = ind == (len(input_output_dimensions) - 1)

            self.upsamplers.append(
                nn.ModuleList(
                    [
                        block_klass(out_dimension + in_dimension, out_dimension, temporal_embedding_dimension=temporal_embedding_dimension),
                        block_klass(out_dimension + in_dimension, out_dimension, temporal_embedding_dimension=temporal_embedding_dimension),
                        Residual(PreNorm(out_dimension, LinearAttention(out_dimension))),
                        upsample(out_dimension, in_dimension)
                        if not is_last
                        else nn.Conv2d(out_dimension, in_dimension, 3, padding=1),
                    ]
                )
            )

        self.output_dimension: int = default(output_dimension, num_channels)

        self.final_res_block = block_klass(input_dimension * 2, input_dimension, temporal_embedding_dimension=temporal_embedding_dimension)
        self.final_conv = nn.Conv2d(input_dimension, self.output_dimension, 1)

    def forward(self, x: Tensor, time: Tensor, x_self_cond=None) -> Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downsamplers:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)


        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)


        for block1, block2, attn, upsample in self.upsamplers:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

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
