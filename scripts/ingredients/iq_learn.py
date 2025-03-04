"""This ingredient provides a reinforcement learning algorithm from stable-baselines3.

The algorithm instance is either freshly constructed or loaded from a file.
"""

import logging
import warnings
from typing import Any, Dict, Mapping, Optional, Type

import sacred
import stable_baselines3 as sb3
from stable_baselines3.common import (
    base_class,
    buffers,
    off_policy_algorithm,
    on_policy_algorithm,
    vec_env,
)

from difo.baselines import IQLearn
from imitation.policies import serialize
from imitation.policies.replay_buffer_wrapper import ReplayBufferRewardWrapper
from imitation.rewards.reward_function import RewardFn
from scripts.ingredients import logging as logging_ingredient
from scripts.ingredients.policy import policy_ingredient

iq_ingredient = sacred.Ingredient(
    "iq",
    ingredients=[policy_ingredient, logging_ingredient.logging_ingredient],
)
logger = logging.getLogger(__name__)


@iq_ingredient.config
def config():
    batch_size = 256
    iq_kwargs = dict(
        div_method="kl",  # kl, kl2, kl_fix, hellinger, js
        loss_method="value",  # value_expert, value, v0
        grad_pen=False,
        lambda_gp=10.0,
        chi=False,
        regularize=False,
        use_target_value=False,
    )
    rl_kwargs = dict(critic_learning_rate=None)
    locals()  # quieten flake8


@iq_ingredient.capture
def make_iq(
    venv: vec_env.VecEnv,
    demonstrations,
    batch_size: int,
    iq_kwargs: Mapping[str, Any],
    rl_kwargs: Mapping[str, Any],
    policy: Mapping[str, Any],
    _seed: int,
) -> IQLearn:
    """Instantiates a Stable Baselines3 RL algorithm.

    Args:
        venv: The vectorized environment to train on.
        rl_cls: Type of a Stable Baselines3 RL algorithm.
        batch_size: The batch size of the RL algorithm.
        rl_kwargs: Keyword arguments for RL algorithm constructor.
        policy: Configuration for the policy ingredient. We need the
            policy_cls and policy_kwargs component.
        relabel_reward_fn: Reward function used for reward relabeling
            in replay or rollout buffers of RL algorithms.

    Returns:
        The RL algorithm.

    Raises:
        ValueError: `gen_batch_size` not divisible by `venv.num_envs`.
        TypeError: `rl_cls` is neither `OnPolicyAlgorithm` nor `OffPolicyAlgorithm`.
    """
    if batch_size % venv.num_envs != 0:
        raise ValueError(
            f"num_envs={venv.num_envs} must evenly divide batch_size={batch_size}.",
        )
    rl_kwargs = dict(rl_kwargs)

    iq_trainer = IQLearn(
        policy=policy["policy_cls"],
        policy_kwargs=dict(policy["policy_kwargs"]),
        env=venv,
        seed=_seed,
        batch_size=batch_size,
        demonstrations=demonstrations,
        **iq_kwargs,
        **rl_kwargs,
    )

    logger.info(f"Policy network summary:\n {iq_trainer.policy}")
    return iq_trainer
