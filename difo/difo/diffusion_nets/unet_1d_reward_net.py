from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import torch as th
from diffusers import DDPMScheduler, SchedulerMixin
from stable_baselines3.common import preprocessing
from torch import nn

from .base import DIFORewardNet
from .unet import UNet1DModel


class UNet1DRewardNet(DIFORewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        # DIFO
        scheduler_cls: SchedulerMixin = DDPMScheduler,
        scheduler_kwargs: Dict[str, Any] = dict(num_train_timesteps=1000),
        sample_strategy: str = "random",
        sample_strategy_kwargs: Dict[str, Any] = dict(low=250, high=750),
        normalize_input_layer: Optional[Type[nn.Module]] = None,
        # UNet2D
        diffusion_net_kwargs: Dict[str, Any] = dict(),
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            sample_strategy=sample_strategy,
            sample_strategy_kwargs=sample_strategy_kwargs,
        )

        input_size = 0
        output_size = 0
        self.use_state = use_state
        if self.use_state:
            input_size += preprocessing.get_flattened_obs_dim(observation_space)
            output_size += preprocessing.get_flattened_obs_dim(observation_space)
        self.use_action = use_action
        if self.use_action:
            input_size += preprocessing.get_flattened_obs_dim(action_space)
            output_size += preprocessing.get_flattened_obs_dim(action_space)
        self.use_next_state = use_next_state
        if self.use_next_state:
            input_size += preprocessing.get_flattened_obs_dim(observation_space)
            output_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.net = UNet1DModel(
            input_size=input_size,
            output_size=output_size,
            **diffusion_net_kwargs,
        )

        if normalize_input_layer is not None:
            self.normalize_input_layer = normalize_input_layer(input_size)
        else:
            self.normalize_input_layer = nn.Identity()

    def _parse_inputs(
        self,
        state,
        action,
        next_state,
        done,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        inputs_concat = th.cat(inputs, dim=1)
        inputs_concat = self.normalize_input_layer(inputs_concat)

        return inputs_concat, None

    def _forward_net(
        self,
        sample,
        timestep,
        class_labels,
        condition_emb,
    ) -> th.Tensor:
        return self.net(
            sample=sample,
            timestep=timestep,
            class_labels=class_labels,
            return_dict=False,
        )[0]
