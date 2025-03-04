from collections import deque
from typing import Callable, Mapping, Optional, Type

import numpy as np
import torch as th
import tqdm

from imitation.algorithms import base
from imitation.rewards import reward_nets
from imitation.util import logger, util

from .diffusion_nets.base import DIFORewardNet
from .reward_fn import DenoisingRewardNetFromMSE


class ExpertLossRewardNet(reward_nets.RewardNet):
    r"""Converts the discriminator logits raw value to a reward signal.

    Wrapper for reward network that takes in the logits of the discriminator
    probability distribution and outputs the corresponding reward for the AIRL
    algorithm.
    """

    def __init__(self, base: DIFORewardNet, logit_scale: float = 10):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base
        self.logit_scale = logit_scale

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        expert_loss = self.base.forward_expert(state, action, next_state, done)
        return expert_loss


class DiffusionTrainer(base.DemonstrationAlgorithm):
    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        reward_net: reward_nets.RewardNet,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping] = None,
        log_interval: int = 100,
    ):
        self.demo_batch_size = demo_batch_size
        self._demo_data_loader = None
        self._endless_expert_iterator = None

        # self.demonstrations = demonstrations
        self.reward_net = ExpertLossRewardNet(reward_net).to("cuda")
        self._optimizer = optimizer_cls(
            self.reward_net.parameters(),
            **(optimizer_kwargs or {}),
        )
        self.log_interval = log_interval

        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations, batch_size=self.demo_batch_size
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        assert self._endless_expert_iterator is not None
        return next(self._endless_expert_iterator)

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return DenoisingRewardNetFromMSE(self.reward_net)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
        log: bool = True,
    ) -> float:
        losses = deque(maxlen=100)
        for step in tqdm.trange(total_timesteps, dynamic_ncols=True):
            batch = self._make_batch()

            loss = self.reward_net(**batch)
            loss = th.mean(loss)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

            if log:
                self.logger.record("diffusion/loss", loss.item())
                self.logger.record("diffusion/step", step)

                if step % self.log_interval == 0:
                    self.logger.dump(step)

            if callback:
                callback(step)

        return np.mean(losses)

    def _make_batch(self) -> Mapping:
        expert_batch = self._next_expert_batch()
        expert_batch = dict(expert_batch)

        obs = expert_batch["obs"]
        acts = expert_batch["acts"]
        next_obs = expert_batch["next_obs"]
        dones = expert_batch["dones"]

        obs_th, acts_th, next_obs_th, dones_th = self.reward_net.preprocess(
            obs,
            acts,
            next_obs,
            dones,
        )

        return {
            "state": obs_th,
            "action": acts_th,
            "next_state": next_obs_th,
            "done": dones_th,
        }
