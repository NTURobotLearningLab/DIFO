"""Generative Adversarial Imitation Learning (GAIL)."""

from typing import Any, Dict, Optional, Tuple, Union

import torch as th
from stable_baselines3.common import base_class, vec_env
from torch.nn import functional as F

from difo.difo.reward_fn import REWARD_FN_DICT, GAILRewardNetFromDiscriminatorLogit
from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


class GAIL(common.AdversarialTrainer):
    """Generative Adversarial Imitation Learning (`GAIL`_).

    .. _GAIL: https://arxiv.org/abs/1606.03476
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        processed_reward_cls: Union[
            reward_nets.RewardNet, str
        ] = GAILRewardNetFromDiscriminatorLogit,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

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
        # Raw self._reward_net is discriminator logits
        reward_net = reward_net.to(gen_algo.device)
        # Process it to produce output suitable for RL training
        # Applies a -log(sigmoid(-logits)) to the logits (see class for explanation)
        if isinstance(processed_reward_cls, str):
            processed_reward_cls = REWARD_FN_DICT[processed_reward_cls]
        self._processed_reward = processed_reward_cls(reward_net)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

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
        r"""Compute the loss of the discriminator for each state-action sample.

        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: whether a `terminal state` (as defined under the MDP of the task) has
                been reached.
            log_policy_act_prob: The log probability of the action taken by the
                generator, :math:`\log{P(a|s)}`.

        Returns:
            loss: The loss of the discriminator for each state-action sample.
            disc_logits: Discriminator logits for each state-action sample.
            info: Additional information for logging.
        """

        disc_logits = self.logits_expert_is_high(
            state, action, next_state, done, log_policy_act_prob
        )
        loss = F.binary_cross_entropy_with_logits(
            disc_logits,
            labels_expert_is_one.float(),
        )
        return loss, disc_logits, {}

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._processed_reward

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._processed_reward
