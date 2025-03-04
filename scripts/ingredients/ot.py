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

from difo.baselines import OT
from imitation.policies import serialize
from imitation.policies.replay_buffer_wrapper import ReplayBufferRewardWrapper
from imitation.rewards.reward_function import RewardFn
from scripts.ingredients import logging as logging_ingredient
from scripts.ingredients.policy import policy_ingredient

ot_ingredient = sacred.Ingredient(
    "ot",
    ingredients=[policy_ingredient, logging_ingredient.logging_ingredient],
)
logger = logging.getLogger(__name__)


@ot_ingredient.config
def config():
    batch_size = 256
    ot_kwargs = dict(
        reward_scale=100.0,
    )
    rl_kwargs = dict()
    locals()  # quieten flake8


@ot_ingredient.capture
def make_ot(
    venv: vec_env.VecEnv,
    demonstrations,
    batch_size: int,
    ot_kwargs: Mapping[str, Any],
    rl_kwargs: Mapping[str, Any],
    policy: Mapping[str, Any],
    _seed: int,
) -> OT:
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

    ot_trainer = OT(
        policy=policy["policy_cls"],
        policy_kwargs=dict(policy["policy_kwargs"]),
        env=venv,
        seed=_seed,
        batch_size=batch_size,
        demonstrations=demonstrations,
        **ot_kwargs,
        **rl_kwargs,
    )

    logger.info(f"Policy network summary:\n {ot_trainer.policy}")
    return ot_trainer
