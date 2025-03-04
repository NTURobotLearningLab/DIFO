from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from ot import sinkhorn
from stable_baselines3.common import preprocessing
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from imitation.rewards import reward_nets
from imitation.rewards.reward_nets import RewardNet
from imitation.util import networks, util


class OTRewardNet(RewardNet):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        emb_size: int = 32,
        sinkhorn_rew_scale: float = 10.0,
        num_expert_samples: int = 10,
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            "out_size": emb_size,
            **kwargs,
            # we do not want the values below to be overridden
            "in_size": combined_size,
        }

        self.critic_net = networks.build_mlp(**full_build_mlp_kwargs)
        self.sinkhorn_rew_scale = sinkhorn_rew_scale
        self.num_expert_samples = num_expert_samples

        self.demonstrations = None

    def set_demonstrations(self, demonstrations):
        self.demonstrations = demonstrations

    def _sample_trajs(self, n_trajs: int):
        if self.demonstrations is None:
            raise ValueError("No demonstrations provided.")
        idxs = np.random.choice(len(self.demonstrations), n_trajs)
        return [self.demonstrations[idx] for idx in idxs]

    def _cosine_distance(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        x = util.safe_to_tensor(x).to(th.float32).to(self.device)
        y = util.safe_to_tensor(y).to(th.float32).to(self.device)
        cost_matrix = th.mm(x, y.T)
        x_norm = th.norm(x, p=2, dim=-1)
        y_norm = th.norm(y, p=2, dim=-1)
        x_n = x_norm.unsqueeze(1)
        y_n = y_norm.unsqueeze(1)
        norms = th.mm(x_n, y_n.T)
        cost_matrix = 1 - cost_matrix / norms
        return cost_matrix

    def optimal_transport_plan(self, X, Y, cost_matrix, niter=100, epsilon=0.01):
        X_pot = np.ones(X.shape[0]) * (1 / X.shape[0])
        Y_pot = np.ones(Y.shape[0]) * (1 / Y.shape[0])
        c_m = cost_matrix.data.detach().cpu().numpy()
        transport_plan = sinkhorn(X_pot, Y_pot, c_m, epsilon, numItermax=niter)
        transport_plan = th.from_numpy(transport_plan).to(th.float32).to(X.device)
        transport_plan.requires_grad = False
        return transport_plan

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        sampled_trajs = self._sample_trajs(self.num_expert_samples)

        agent_inputs = []
        if self.use_state:
            agent_inputs.append(state)
        if self.use_action:
            agent_inputs.append(action)
        if self.use_next_state:
            agent_inputs.append(next_state)
        if self.use_done:
            agent_inputs.append(done)
        agent_inputs = th.cat(agent_inputs, dim=-1)

        ot_rewards_list = []
        distance_list = []
        for traj in sampled_trajs:
            expert_inputs = []
            if self.use_state:
                expert_inputs.append(util.safe_to_tensor(traj.obs[:-1]))
            if self.use_action:
                expert_inputs.append(util.safe_to_tensor(traj.acts))
            if self.use_next_state:
                expert_inputs.append(util.safe_to_tensor(traj.obs[1:]))
            expert_inputs = th.cat(expert_inputs, dim=-1)

            cost_matrix = self._cosine_distance(agent_inputs, expert_inputs)
            transport_plan = self.optimal_transport_plan(
                agent_inputs, expert_inputs, cost_matrix
            )
            costs = th.diag(th.mm(transport_plan, cost_matrix.T))
            distance = th.sum(costs)
            ot_rewards = -costs * self.sinkhorn_rew_scale

            distance_list.append(distance)
            ot_rewards_list.append(ot_rewards)
        closest_demo_index = th.argmin(th.stack(distance_list))
        ot_rewards = ot_rewards_list[closest_demo_index]
        return ot_rewards
