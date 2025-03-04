"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import json
import logging
import os.path as osp
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy as np
import sacred.commands
from sacred.observers import FileStorageObserver
from stable_baselines3 import PPO, SAC

import wandb
from imitation.data import rollout, types
from imitation.util import util
from scripts.config.train_imitation import train_imitation_ex
from scripts.ingredients import (
    bc as bc_ingredient,
    bco as bco_ingredient,
    demonstrations,
    environment,
    expert,
    iq_learn as iq_ingredient,
    logging as logging_ingredient,
    ot as ot_ingredient,
    policy_evaluation,
)

logger = logging.getLogger(__name__)


def _all_trajectories_have_reward(trajectories: Sequence[types.Trajectory]) -> bool:
    """Returns True if all trajectories have reward information."""
    return all(isinstance(t, types.TrajectoryWithRew) for t in trajectories)


def _try_computing_expert_stats(
    expert_trajs: Sequence[types.Trajectory],
) -> Optional[Mapping[str, float]]:
    """Adds expert statistics to `stats` if all expert trajectories have reward."""
    if _all_trajectories_have_reward(expert_trajs):
        return rollout.rollout_stats(
            cast(Sequence[types.TrajectoryWithRew], expert_trajs),
        )
    else:
        logger.warning(
            "Expert trajectories do not have reward information, so expert "
            "statistics cannot be computed.",
        )
        return None


def _collect_stats(
    imit_stats: Mapping[str, float],
    expert_trajs: Sequence[types.Trajectory],
) -> Mapping[str, Mapping[str, Any]]:
    stats = {"imit_stats": imit_stats}
    expert_stats = _try_computing_expert_stats(expert_trajs)
    if expert_stats is not None:
        stats["expert_stats"] = expert_stats

    return stats


@train_imitation_ex.command
def bc(
    bc: Dict[str, Any],
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs BC training.

    Args:
        bc: Configuration for BC training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs = demonstrations.get_expert_trajectories()

    render_mode = "rgb_array"
    with environment.make_venv(  # type: ignore[wrong-keyword-args]
        env_make_kwargs=dict(render_mode=render_mode),
    ) as venv:
        bc_trainer = bc_ingredient.make_bc(venv, expert_trajs, custom_logger)

        bc_train_kwargs = dict(log_rollouts_venv=venv, **bc["train_kwargs"])
        if bc_train_kwargs["n_epochs"] is None and bc_train_kwargs["n_batches"] is None:
            bc_train_kwargs["n_batches"] = 50_000

        bc_trainer.train(**bc_train_kwargs)
        # TODO(adam): add checkpointing to BC?
        # util.save_policy(bc_trainer.policy, policy_path=osp.join(log_dir, "final.th"))

        # Save as PPO model for evaluation
        policy_cls = _run.config["policy"]["policy_cls"]
        policy_kwargs = _run.config["policy"]["policy_kwargs"]
        model = PPO(
            policy=policy_cls,
            env=venv,
            policy_kwargs=policy_kwargs,
        )
        model.policy = bc_trainer.policy
        model.save(osp.join(log_dir, "model"))

        logger.info("Evaluating policy...")
        imit_stats = policy_evaluation.eval_policy(bc_trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)
    json.dump(stats, open(osp.join(log_dir, "stats.json"), "w"), indent=2)

    wandb_stats = {f"eval_stats/{k}": v for k, v in imit_stats.items()}
    if wandb.run is not None:
        wandb.log(wandb_stats)

    return stats


@train_imitation_ex.command
def iq_learn(
    iq: Dict[str, Any],
    total_timesteps,
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs IQ-Learn training.

    Args:
        bc: Configuration for IQ-Learn training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    print(iq)
    sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs = demonstrations.get_expert_trajectories()

    render_mode = "rgb_array"
    with environment.make_venv(  # type: ignore[wrong-keyword-args]
        env_make_kwargs=dict(render_mode=render_mode),
    ) as venv:
        trainer = iq_ingredient.make_iq(venv, demonstrations=expert_trajs)
        trainer = cast(SAC, trainer)
        trainer.set_logger(custom_logger)

        trainer.learn(total_timesteps, progress_bar=True)

        logger.info("Evaluating policy...")
        imit_stats = policy_evaluation.eval_policy(trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)
    json.dump(stats, open(osp.join(log_dir, "stats.json"), "w"), indent=2)

    wandb_stats = {f"eval_stats/{k}": v for k, v in imit_stats.items()}
    if wandb.run is not None:
        wandb.log(wandb_stats)

    return stats


@train_imitation_ex.command
def ot(
    ot: Dict[str, Any],
    total_timesteps,
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs IQ-Learn training.

    Args:
        bc: Configuration for IQ-Learn training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    print(ot)
    sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs = demonstrations.get_expert_trajectories()

    render_mode = "rgb_array"
    with environment.make_venv(  # type: ignore[wrong-keyword-args]
        env_make_kwargs=dict(render_mode=render_mode),
    ) as venv:
        trainer = ot_ingredient.make_ot(venv, demonstrations=expert_trajs)
        trainer = cast(SAC, trainer)
        trainer.set_logger(custom_logger)

        trainer.learn(total_timesteps, progress_bar=True)

        logger.info("Evaluating policy...")
        imit_stats = policy_evaluation.eval_policy(trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)
    json.dump(stats, open(osp.join(log_dir, "stats.json"), "w"), indent=2)

    wandb_stats = {f"eval_stats/{k}": v for k, v in imit_stats.items()}
    if wandb.run is not None:
        wandb.log(wandb_stats)

    return stats


@train_imitation_ex.command
def bco(
    bco: Dict[str, Any],
    total_timesteps: Optional[int],
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs BC training.

    Args:
        bc: Configuration for BC training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs = demonstrations.get_expert_trajectories()
    with environment.make_venv() as venv:  # type: ignore[wrong-arg-count]
        bco_trainer = bco_ingredient.make_bco(
            venv, expert_trajs, custom_logger, save_dir=log_dir, **bco
        )
        bco_train_kwargs = dict(
            total_timesteps=total_timesteps,
            log_rollouts_venv=venv,
            **bco["train_kwargs"],
        )
        # if (
        #     bco_train_kwargs["bc_n_epochs"] is None
        #     and bco_train_kwargs["bc_n_batches"] is None
        # ):
        #     bco_train_kwargs["bc_n_batches"] = 50_000

        bco_trainer.train(**bco_train_kwargs)
        # TODO(adam): add checkpointing to BC?
        util.save_policy(bco_trainer.policy, policy_path=osp.join(log_dir, "final.th"))
        imit_stats = policy_evaluation.eval_policy(bco_trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)
    json.dump(stats, open(osp.join(log_dir, "stats.json"), "w"), indent=2)

    wandb_stats = {f"eval_stats/{k}": v for k, v in imit_stats.items()}
    if wandb.run is not None:
        wandb.log(wandb_stats)

    return stats


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_imitation"
    observer = FileStorageObserver(observer_path)
    train_imitation_ex.observers.append(observer)
    train_imitation_ex.run_commandline()


if __name__ == "__main__":
    main_console()
