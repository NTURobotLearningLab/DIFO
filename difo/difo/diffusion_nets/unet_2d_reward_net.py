from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch as th
from diffusers import DDPMScheduler, SchedulerMixin, UNet2DModel
from torchvision.transforms.v2 import Compose, Normalize, Resize

from .base import DIFORewardNet


class UNet2DRewardNet(DIFORewardNet):
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
        # UNet2D
        diffusion_input_size: int = 64,
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

        if use_action or use_done:
            raise NotImplementedError(
                "use_action and use_done not implemented for UNet2DRewardNet"
            )

        # get image size and channel
        if observation_space.shape[0] == observation_space.shape[1]:
            image_channel = observation_space.shape[2]
            self.hwc_format = True
        else:
            image_channel = observation_space.shape[0]
            self.hwc_format = False

        # get in_channels
        in_channels = 2 * image_channel
        out_channels = 2 * image_channel

        self.transform = Compose(
            [
                Resize(diffusion_input_size),
                Normalize([0.5], [0.5]),
            ]
        )
        self.net = UNet2DModel(
            sample_size=diffusion_input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_class_embeds=2,
            **diffusion_net_kwargs,
        )

    def _parse_inputs(
        self,
        state,
        action,
        next_state,
        done,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        inputs = [state, next_state]
        if self.hwc_format:
            inputs = [x.permute(0, 3, 1, 2) for x in inputs]
        inputs_concat = th.cat(inputs, dim=1)
        inputs_concat = self.transform(inputs_concat)

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
