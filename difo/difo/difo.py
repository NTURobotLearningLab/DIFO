"""Diffusion Imitation from Observation (DIFO)."""

from typing import Optional, Union

import torch as th
from stable_baselines3.common import base_class, vec_env
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets

from .diffusion_nets import DIFORewardNet
from .reward_fn import REWARD_FN_DICT, GAILRewardNetFromDiscriminatorLogit


class DIFORewardNetFromLosses(reward_nets.RewardNet):
    r"""Converts the discriminator logits raw value to a reward signal.

    Wrapper for reward network that takes in the logits of the discriminator
    probability distribution and outputs the corresponding reward for the AIRL
    algorithm.
    """

    def __init__(self, base: reward_nets.RewardNet, logit_scale: float = 10):
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
        generator_loss, expert_loss = self.base.forward(state, action, next_state, done)
        rewards = (generator_loss - expert_loss) * self.logit_scale
        n_sample = getattr(self.base, "n_sample", 1)
        if n_sample == 1:
            return rewards.squeeze()
        return rewards.reshape(n_sample, -1).mean(dim=0).squeeze()


class DIFO(common.AdversarialTrainer):
    """Diffusion Imitation from Observation (`DIFO`_)."""

    _reward_net: DIFORewardNet

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: DIFORewardNet,
        processed_reward_cls: Union[
            reward_nets.RewardNet, str
        ] = GAILRewardNetFromDiscriminatorLogit,
        mse_positive_weight: float = 1,
        mse_negative_weight: float = 0,
        bce_weight: float = 1,
        mse_weight: float = 0,
        agent_weight: float = 0,
        expert_weight: float = 1,
        logit_scale: float = 100,
        **kwargs,
    ):
        """Diffusion Rewards Guided Adversarial Imitation Learning.

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
        reward_net = reward_net.to(gen_algo.device)
        # Process it to produce output suitable for RL training
        # Applies a -log(sigmoid(-logits)) to the logits (see class for explanation)
        if isinstance(processed_reward_cls, str):
            processed_reward_cls = REWARD_FN_DICT[processed_reward_cls]
        self._processed_reward = processed_reward_cls(
            DIFORewardNetFromLosses(reward_net)
        )
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

        self.mse_positive_weight = mse_positive_weight
        self.mse_negative_weight = mse_negative_weight
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.agent_weight = agent_weight
        self.expert_weight = expert_weight
        self.logit_scale = logit_scale

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
        generator_loss, expert_loss = self._reward_net(state, action, next_state, done)
        assert generator_loss.shape == expert_loss.shape == state.shape[:1]
        return generator_loss, expert_loss

    def calc_loss(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
        labels_expert_is_one: th.Tensor = None,
    ) -> th.Tensor:
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
        del log_policy_act_prob

        bin_is_generated_true = labels_expert_is_one == 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)

        generator_loss, expert_loss = self._reward_net(state, action, next_state, done)
        disc_logits = (generator_loss - expert_loss) * self.logit_scale

        # BCE
        if self.bce_weight != 0:
            bce_loss = F.binary_cross_entropy_with_logits(
                disc_logits, labels_expert_is_one.float()
            )
        else:
            bce_loss = th.tensor(0.0, device=state.device)

        # MSE
        if self.mse_weight != 0 or (
            self.mse_positive_weight != 0 and self.mse_negative_weight != 0
        ):
            mse_loss = th.zeros_like(disc_logits)
            mse_loss[bin_is_generated_true] = (
                generator_loss[bin_is_generated_true] * self.mse_positive_weight
                - expert_loss[bin_is_generated_true] * self.mse_negative_weight
            ) * self.agent_weight
            mse_loss[bin_is_expert_true] = (
                expert_loss[bin_is_expert_true] * self.mse_positive_weight
                - generator_loss[bin_is_expert_true] * self.mse_negative_weight
            ) * self.expert_weight
            mse_loss = mse_loss.mean()
        else:
            mse_loss = th.tensor(0.0, device=state.device)

        loss = mse_loss * self.mse_weight + bce_loss * self.bce_weight

        info = {
            "bce_loss": bce_loss,
            "mse_loss": mse_loss,
            "gen_cond_gen_loss_mean": generator_loss[bin_is_generated_true].mean(),
            "gen_cond_exp_loss_mean": expert_loss[bin_is_generated_true].mean(),
            "exp_cond_gen_loss_mean": generator_loss[bin_is_expert_true].mean(),
            "exp_cond_exp_loss_mean": expert_loss[bin_is_expert_true].mean(),
        }

        return loss, disc_logits, info

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._processed_reward

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._processed_reward
