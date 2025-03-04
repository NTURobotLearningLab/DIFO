from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch as th
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput
from einops import rearrange
from torch import nn


def exists(x):
    return x is not None


@dataclass
class UNet1DOutput(BaseOutput):
    """
    The output of [`UNet1DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim * 2),
        )
        self.relu = nn.ReLU()

    def forward(self, x, emb):
        x = self.block(x)
        emb = self.emb_mlp(emb)
        shift, scale = emb.chunk(2, dim=-1)
        x = x * scale + shift
        x = self.relu(x)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(self.theta) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class UNet1DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_size: int,
        output_size: int,
        emb_dim: Optional[int] = None,
        num_units: Iterable[int] = [64, 64, 64, 64],
        concat_emb: bool = False,
    ):
        super().__init__()

        if emb_dim is None:
            emb_dim = input_size * 4
        self.label_emb = nn.Embedding(2, emb_dim)
        self.time_emb = SinusoidalPosEmb(emb_dim)
        self.concat_emb = concat_emb

        self.init_linear = nn.Linear(input_size, num_units[0])

        self.down_blocks = nn.ModuleList()
        for i in range(1, len(num_units)):
            self.down_blocks.append(
                ResidualBlock(
                    num_units[i - 1],
                    num_units[i],
                    emb_dim * 2 if concat_emb else emb_dim,
                )
            )

        self.mid_mlp = nn.Sequential(
            nn.Linear(num_units[-1], num_units[-1] // 2),
            nn.ReLU(),
            nn.Linear(num_units[-1] // 2, num_units[-1]),
        )

        num_units = num_units[::-1]
        self.up_blocks = nn.ModuleList()
        for i in range(len(num_units) - 1):
            self.up_blocks.append(
                ResidualBlock(
                    num_units[i] * 2,
                    num_units[i + 1],
                    emb_dim * 2 if concat_emb else emb_dim,
                )
            )

        self.final_linear = nn.Linear(num_units[-1], output_size)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        condition_emb: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple]:
        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # embeddings
        t_embedding = self.time_emb(timesteps)
        label_embedding = self.label_emb(class_labels)
        if self.concat_emb:
            embedding = th.cat([t_embedding, label_embedding], dim=-1)
        else:
            embedding = t_embedding + label_embedding
        if exists(condition_emb):
            embedding = embedding + condition_emb

        sample = self.init_linear(sample)

        # down
        feats = []
        for down_block in self.down_blocks:
            sample = down_block(sample, embedding)
            feats.append(sample)

        # mid
        sample = self.mid_mlp(sample)

        # up
        feats = feats[::-1]
        for up_block, feat in zip(self.up_blocks, feats):
            sample = th.cat([sample, feat], dim=-1)
            sample = up_block(sample, embedding)

        sample = self.final_linear(sample)

        if not return_dict:
            return (sample,)

        return UNet1DOutput(sample=sample)
