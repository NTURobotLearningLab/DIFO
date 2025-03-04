import abc
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from diffusers import ModelMixin, SchedulerMixin
from einops import reduce, repeat
from torch import nn

from imitation.rewards.reward_nets import RewardNet

from .pipeline import DDPMConditionPipeline


class DIFORewardNet(RewardNet):
    net: ModelMixin
    noise_scheduler: SchedulerMixin
    normalize_input_layer: nn.Module

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool,
        use_action: bool,
        use_next_state: bool,
        use_done: bool,
        # DIFO
        scheduler_cls: SchedulerMixin,
        scheduler_kwargs: Dict[str, Any],
        sample_strategy: str,
        sample_strategy_kwargs: Dict[str, Any],
    ):
        super().__init__(observation_space, action_space)

        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done

        self.n_steps = scheduler_kwargs["num_train_timesteps"]
        self.noise_scheduler = scheduler_cls(**scheduler_kwargs)

        # sample strategy
        self.sample_strategy = sample_strategy
        self.sample_strategy_kwargs = sample_strategy_kwargs
        assert self.sample_strategy in ["random", "constant", "symmetry"]
        self.n_sample = sample_strategy_kwargs.get("n_sample", 1)
        if self.sample_strategy == "random":
            if "low" not in self.sample_strategy_kwargs:
                UserWarning("low not in sample_strategy_kwargs. Set low to 0")
            if "high" not in self.sample_strategy_kwargs:
                UserWarning(
                    f"high not in sample_strategy_kwargs. Set high to {self.n_steps}"
                )
            if "n_sample" not in self.sample_strategy_kwargs:
                UserWarning("n_sample not in sample_strategy_kwargs. Set n_sample to 1")
        elif self.sample_strategy == "constant":
            assert "step" in self.sample_strategy_kwargs
            if self.sample_strategy_kwargs["step"] >= self.n_steps:
                UserWarning("Step is larger than n_steps. Set step to n_steps - 1")

    def _sample_time(self, batch_size, sample_strategy, sample_strategy_kwargs):
        if sample_strategy == "random":
            low = sample_strategy_kwargs.get("low", 0)
            high = sample_strategy_kwargs.get("high", self.n_steps)
            t = th.randint(low, high, size=(batch_size,), device=self.device)
        elif sample_strategy == "constant":
            step = sample_strategy_kwargs["step"]
            if step >= self.n_steps:
                step = self.n_steps - 1
            t = th.full((batch_size,), step, device=self.device)
        elif sample_strategy == "symmetry":
            t = th.randint(
                0, self.n_steps, size=(batch_size // 2,), device=self.device
            )  # [batch_size // 2]
            t = th.cat([t, self.n_steps - 1 - t], dim=0)  # [batch_size]
        return t

    @abc.abstractmethod
    def _parse_inputs(
        self,
        state,
        action,
        next_state,
        done,
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        pass

    @abc.abstractmethod
    def _forward_net(
        self,
        sample,
        timestep,
        class_labels,
        condition_emb,
    ) -> th.Tensor:
        pass

    def _diffusion_loss(
        self,
        clean_data: th.Tensor,
        condition_label: int,
        sample_strategy: str,
        sample_strategy_kwargs: Dict[str, Any],
        condition_emb: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Calculate diffusion loss.

        Args:
            - clean_data (torch.Tensor): clean data tensor.
            - condition (int): condition for the diffusion process.
            - sample_strategy (str): sample strategy for the diffusion process.
            - sample_strategy_kwargs (Dict[str, Any]): sample strategy kwargs.

        Returns:
            - loss (torch.Tensor): Diffusion loss tensor.

        """

        n_sample = sample_strategy_kwargs.get("n_sample", 1)
        clean_data = repeat(clean_data, "b ... -> (n_sample b) ...", n_sample=n_sample)
        if condition_emb is not None:
            condition_emb = condition_emb.repeat(n_sample, 1)
        batch_size = clean_data.shape[0]
        # add noise
        noise = th.randn(clean_data.shape, device=clean_data.device)
        timesteps = self._sample_time(
            batch_size, sample_strategy, sample_strategy_kwargs
        )
        noisy_data = self.noise_scheduler.add_noise(clean_data, noise, timesteps)

        # condition input
        condition_input = th.full(
            (batch_size,), condition_label, device=self.device, dtype=th.int
        )

        # get noise prediction
        noise_pred = self._forward_net(
            noisy_data,
            timesteps,
            condition_input,
            condition_emb,
        )

        # calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        return loss

    def forward(
        self,
        state,
        action,
        next_state,
        done,
    ):
        # random sample strategy during training
        if self.training:
            sample_strategy = "random"
            sample_strategy_kwargs = dict()
        else:
            sample_strategy = self.sample_strategy
            sample_strategy_kwargs = self.sample_strategy_kwargs

        # get inputs
        inputs, conditions = self._parse_inputs(state, action, next_state, done)

        # calculate loss
        expert_loss = self._diffusion_loss(
            inputs,
            1,
            sample_strategy,
            sample_strategy_kwargs,
            conditions,
        )
        generator_loss = self._diffusion_loss(
            inputs,
            0,
            sample_strategy,
            sample_strategy_kwargs,
            conditions,
        )

        return generator_loss, expert_loss

    def forward_expert(
        self,
        state,
        action,
        next_state,
        done,
    ):
        # random sample strategy during training
        if self.training:
            sample_strategy = "random"
            sample_strategy_kwargs = dict()
        else:
            sample_strategy = self.sample_strategy
            sample_strategy_kwargs = self.sample_strategy_kwargs

        # get inputs
        inputs, conditions = self._parse_inputs(state, action, next_state, done)

        # calculate loss
        expert_loss = self._diffusion_loss(
            inputs,
            1,
            sample_strategy,
            sample_strategy_kwargs,
            conditions,
        )

        return expert_loss

    @th.inference_mode
    def sample(self, batch_size, condition, state_condition=None) -> th.Tensor:
        pipeline = DDPMConditionPipeline(
            unet=self.net, scheduler=self.noise_scheduler
        ).to(self.device)

        if state_condition is not None:
            state_condition = (
                th.from_numpy(state_condition)
                .unsqueeze(0)
                .to(self.device)
                .to(th.float32)
            )

        _, condition_embs = self._parse_inputs(
            state_condition, state_condition, state_condition, state_condition
        )

        images = pipeline(
            batch_size=batch_size,
            condition=condition,
            condition_embs=condition_embs,
            output_type="th",
        ).images

        if not isinstance(self.normalize_input_layer, nn.Identity):
            images = images.to(self.device)
            images = self.normalize_input_layer.unnormalize(images)
            images = images.cpu()

        return images
