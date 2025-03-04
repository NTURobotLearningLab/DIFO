"""Wasserstein Adversarial Imitation Learning (WAIL)."""

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch as th

# from scipy.stats import wasserstein_distance
from stable_baselines3.common import base_class, vec_env
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


def wass_grad_pen(
    state: th.Tensor,
    action: th.Tensor,
    next_state: th.Tensor,
    done: th.Tensor,
    labels_expert_is_one: th.Tensor,
    disc_fn,
    use_actions: bool = True,
    use_next_state: bool = True,
):
    """Wasserstein Gradient Penalty: https://arxiv.org/pdf/1704.00028"""
    expert_state = state[labels_expert_is_one == 1]
    expert_action = action[labels_expert_is_one == 1]
    expert_next_state = next_state[labels_expert_is_one == 1]

    agent_state = state[labels_expert_is_one == 0]
    agent_action = action[labels_expert_is_one == 0]
    agent_next_state = next_state[labels_expert_is_one == 0]

    grad_pen = th.tensor(0.0, device=state.device)
    return (
        grad_pen,
        (expert_state, expert_action, expert_next_state),
        (agent_state, agent_action, agent_next_state),
    )


class WAIL(common.AdversarialTrainer):
    """Wasserstein Adversarial Imitation Learning (`WAIL`_).

    .. _WAIL: https://arxiv.org/pdf/1906.08113
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        regularize_epsilon: float = 1e-3,
        disc_grad_clip: float = 0.1,
        wass_grad_penalty_weight: float = 10.0,
        **kwargs,
    ):
        """Wasserstein Adversarial Imitation Learning.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensor as input, then computes the logits.
                Used as the GAIL discriminator.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.
        """
        assert (
            disc_grad_clip * wass_grad_penalty_weight == 0
        ), "clip and wass_grad_penalty_weight can only choose one"
        # Raw self._reward_net is discriminator logits
        reward_net = reward_net.to(gen_algo.device)
        self.regularize_epsilon = regularize_epsilon

        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

        self.wass_grad_penalty_weight = wass_grad_penalty_weight

        # unwrap
        while not hasattr(reward_net, "use_state"):
            reward_net = reward_net.base
        self.use_state = reward_net.use_state
        self.use_action = reward_net.use_action
        self.use_next_state = reward_net.use_next_state
        self.use_done = reward_net.use_done

    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        r"""Compute the discriminator's logits for each state-action sample.

        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: whether a `terminal state` (as defined under the MDP of the task) has
                been reached.
            log_policy_act_prob: The log probability of the action taken by the
                generator, :math:`\log{P(a|s)}`.

        Returns:
            The logits of the discriminator for each state-action sample.
        """
        del log_policy_act_prob
        logits = self._reward_net(state, action, next_state, done)
        assert logits.shape == state.shape[:1]
        return logits

    def calc_loss(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
        labels_expert_is_one: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, Any]]:
        disc_logits = self.logits_expert_is_high(
            state, action, next_state, done, log_policy_act_prob
        )

        expert_logits = disc_logits[labels_expert_is_one == 1]
        agent_logits = disc_logits[labels_expert_is_one == 0]

        wass_d, expert_data, agent_data = self.compute_gradient_penalty(
            state, action, next_state, done, labels_expert_is_one
        )

        expert_states, expert_actions, expert_next_states = expert_data
        agent_states, agent_actions, agent_next_states = agent_data
        # wass_d = wasserstein_distance(expert_logits, agent_logits)
        diff = (expert_states - agent_states).flatten(1)
        if self.use_action:
            diff = th.cat([diff, (expert_actions - agent_actions).flatten(1)], dim=-1)
        if self.use_next_state:
            diff = th.cat(
                [diff, (expert_next_states - agent_next_states).flatten(1)], dim=-1
            )
        regularize_distant = th.norm(diff, dim=1, keepdim=True)

        if self.regularize_epsilon != 0.0:
            regularize_term = th.mean(
                ((expert_logits - agent_logits - regularize_distant) ** 2)
                / (4 * self.regularize_epsilon)
            )
        else:
            regularize_term = 0.0

        self.logger.record("agent_loss", th.mean(agent_logits).item())
        self.logger.record("expert_loss", th.mean(expert_logits).item())
        self.logger.record(
            "wass_grad_penalty", self.wass_grad_penalty_weight * wass_d.item()
        )
        self.logger.record("regularize_term", regularize_term.item())

        loss = (
            th.mean(agent_logits - expert_logits)
            + self.wass_grad_penalty_weight * wass_d
            + regularize_term
        )
        return loss, disc_logits, {}

    def compute_gradient_penalty(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        labels_expert_is_one: th.Tensor = None,
    ):
        return wass_grad_pen(
            state,
            action,
            next_state,
            done,
            labels_expert_is_one,
            self.logits_expert_is_high,
            use_actions=self.use_action,
            use_next_state=self.use_next_state,
        )

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._reward_net
