from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch as th
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput
from torch import nn


@dataclass
class MLP1DOutput(BaseOutput):
    """
    The output of [`MLP1DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


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
        # emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP1DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_units: Iterable[int] = [64, 64, 64, 64],
        emb_dim: Optional[int] = None,
    ):
        super().__init__()

        if emb_dim is None:
            emb_dim = 8
        self.label_emb = nn.Embedding(2, emb_dim)
        self.time_emb = SinusoidalPosEmb(emb_dim)

        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.init_layer = nn.Sequential(
            nn.Linear(input_size + emb_dim, num_units[0]),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_units[i], num_units[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(num_units) - 1)
            ]
        )

        self.final_linear = nn.Linear(num_units[-1], output_size)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[MLP1DOutput, Tuple]:
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
        embedding = t_embedding + label_embedding

        sample = th.cat((sample, embedding), dim=-1)
        sample = self.init_layer(sample)

        for layer in self.layers:
            sample = layer(sample)

        sample = self.final_linear(sample)

        if not return_dict:
            return (sample,)

        return MLP1DOutput(sample=sample)
