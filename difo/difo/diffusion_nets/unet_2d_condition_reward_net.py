from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch as th
from diffusers import DDPMScheduler, SchedulerMixin
from stable_baselines3.common.policies import NatureCNN
from torchvision.transforms.v2 import Compose, Normalize, RandomResizedCrop, Resize

from .base import DIFORewardNet
from .unet import UNet2DConditionModel


class UNet2DConditionRewardNet(DIFORewardNet):
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
        diffusion_input_size: Optional[int] = None,
        state_augmentation_size: Optional[int] = None,
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
        image_size = observation_space.shape[1]
        if observation_space.shape[0] == observation_space.shape[1]:
            image_channel = observation_space.shape[2]
            self.hwc_format = True
        else:
            image_channel = observation_space.shape[0]
            self.hwc_format = False

        # get in_channels
        in_channels = 1
        out_channels = 1

        self.transform = []
        if diffusion_input_size is not None:
            self.transform.append(Resize(diffusion_input_size))
        else:
            diffusion_input_size = image_size
        self.transform.append(Normalize([0.5], [0.5]))
        self.transform = Compose(self.transform)

        if state_augmentation_size is not None:
            augmentation_size = state_augmentation_size
        else:
            augmentation_size = diffusion_input_size - 10
        self.condition_transform = []
        self.condition_transform.append(RandomResizedCrop(augmentation_size))
        # self.condition_transform.append(ColorJitter(brightness=0.5, contrast=0.5))
        self.condition_transform.append(Normalize([0.5], [0.5]))
        self.condition_transform = Compose(self.condition_transform)

        self.net = UNet2DConditionModel(
            sample_size=diffusion_input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_class_embeds=2,
            addition_embed_type="image",
            **diffusion_net_kwargs,
        )
        self.state_cnn = NatureCNN(
            gym.spaces.Box(
                low=0,
                high=1,
                shape=(image_channel, augmentation_size, augmentation_size),
            ),
            features_dim=diffusion_net_kwargs.get("encoder_hid_dim"),
            normalized_image=True,
        )

    def _parse_inputs(
        self,
        state: th.Tensor,
        action,
        next_state: th.Tensor,
        done,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        inputs = next_state
        if self.hwc_format:
            inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs[:, -1].unsqueeze(1)
        inputs = self.transform(inputs)

        conditions = state
        if self.hwc_format:
            conditions = conditions.permute(0, 3, 1, 2)
        conditions = self.condition_transform(conditions)
        conditions = self.state_cnn(conditions)

        return inputs, conditions

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
            added_cond_kwargs=dict(image_embeds=condition_emb),
            return_dict=False,
        )[0]
