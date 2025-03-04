"""This ingredient provides BCO algorithm instance.

It is either loaded from disk or constructed from scratch.
"""

import warnings
from typing import Optional, Sequence

import sacred
import torch as th
from stable_baselines3.common import vec_env

from difo.baselines import bco
from imitation.data import types
from scripts.ingredients import policy

bco_ingredient = sacred.Ingredient("bco", ingredients=[policy.policy_ingredient])


@bco_ingredient.config
def config():
    batch_size = 32
    l2_weight = 3e-3  # L2 regularization weight
    ent_weight = 3e-3  # Entropy regularization weight
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(
        lr=4e-4,
    )
    bco_inv_lr: float = 1e-4
    bco_inv_batch_size: int = 32
    transition_buffer_size: int = 1_000_000

    train_kwargs = dict(
        bco_alpha_T=20,
        bco_alpha=0.1,
        bco_inv_steps=10_000,
        bc_n_epochs=1,  # Number of bc epochs per BCO run
        bc_n_batches=None,  # Number of inv batches per BCO run
        log_interval=500,  # Number of updates between Tensorboard/stdout logs
    )
    agent_path = None  # Path to serialized policy. If None, a new policy is created.
    locals()  # quieten flake8 unused variable warning


@bco_ingredient.capture
def make_bco(
    venv: vec_env.VecEnv,
    expert_trajs: Sequence[types.Trajectory],
    custom_logger,
    batch_size: int,
    l2_weight: float,
    ent_weight: float,
    optimizer_cls,
    optimizer_kwargs,
    _rnd,
    bco_inv_lr: float = 1e-4,
    bco_inv_batch_size: int = 32,
    save_dir: Optional[str] = None,
    **kwargs,
) -> bco.BCO:
    return bco.BCO(
        venv=venv,
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=make_or_load_policy(venv),
        bco_inv_lr=bco_inv_lr,
        bco_inv_batch_size=bco_inv_batch_size,
        save_dir=save_dir,
        demonstrations=expert_trajs,
        batch_size=batch_size,
        custom_logger=custom_logger,
        rng=_rnd,
        l2_weight=l2_weight,
        ent_weight=ent_weight,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
    )


@bco_ingredient.capture
def make_or_load_policy(venv: vec_env.VecEnv, agent_path: Optional[str] = None):
    """Makes a policy or loads a policy from a path if provided.

    Args:
        venv: Vectorized environment we will be imitating demos from.
        agent_path: Path to serialized policy. If provided, then load the
            policy from this path. Otherwise, make a new policy.
            Specify only if policy_cls and policy_kwargs are not specified.

    Returns:
        A Stable Baselines3 policy.
    """
    if agent_path is None:
        return policy.make_policy(venv)
    else:
        warnings.warn(
            "When agent_path is specified, policy.policy_cls and policy.policy_kwargs "
            "are ignored.",
            RuntimeWarning,
        )
        return bco.reconstruct_policy(agent_path)
