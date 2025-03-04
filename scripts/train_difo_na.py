import logging
import pathlib
from typing import Mapping, Optional

import numpy as np
import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize

import imitation.policies.serialize as policies_serialize
import wandb
from difo.difo import DiffusionTrainer
from imitation.data import wrappers
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from scripts.config.train_difo_na import train_difo_na_ex
from scripts.ingredients import (
    demonstrations,
    diffusion,
    environment,
    logging as logging_ingredient,
    policy_evaluation,
    reward,
    rl,
)


def save(trainer: DiffusionTrainer, save_path: pathlib.Path):
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_test, save_path / "reward_test.pt")
    logging.info(f"Saved reward_test to {save_path / 'reward_test.pt'}")


@train_difo_na_ex.main
def train_difo_na(
    *,
    _run,
    diffusion_total_timesteps: int,
    total_timesteps: int,
    normalize_reward: bool,
    normalize_kwargs: dict,
    policy_save_interval: int,
    policy_save_final: bool,
    diffusion_save_interval: int,
    diffusion_save_final: bool,
    agent_path: Optional[str],
    _rnd: np.random.Generator,
) -> Mapping[str, float]:
    """Trains an expert policy from scratch and saves the rollouts and policy.

    Checkpoints:
      At applicable training steps `step` (where step is either an integer or
      "final"):

        - Policies are saved to `{log_dir}/policies/{step}/`.
        - Rollouts are saved to `{log_dir}/rollouts/{step}.npz`.

    Args:
        total_timesteps: Number of training timesteps in `model.learn()`.
        normalize_reward: Applies normalization and clipping to the reward function by
            keeping a running average of training rewards. Note: this is may be
            redundant if using a learned reward that is already normalized.
        normalize_kwargs: kwargs for `VecNormalize`.
        reward_type: If provided, then load the serialized reward of this type,
            wrapping the environment in this reward. This is useful to test
            whether a reward model transfers. For more information, see
            `imitation.rewards.serialize.load_reward`.
        reward_path: A specifier, such as a path to a file on disk, used by
            reward_type to load the reward model. For more information, see
            `imitation.rewards.serialize.load_reward`.
        load_reward_kwargs: Additional kwargs to pass to `predict_processed`.
            Examples are 'alpha' for :class: `AddSTDRewardWrapper` and 'update_stats'
            for :class: `NormalizedRewardNet`.
        rollout_save_final: If True, then save rollouts right after training is
            finished.
        rollout_save_n_timesteps: The minimum number of timesteps saved in every
            file. Could be more than `rollout_save_n_timesteps` because
            trajectories are saved by episode rather than by transition.
            Must set exactly one of `rollout_save_n_timesteps`
            and `rollout_save_n_episodes`.
        rollout_save_n_episodes: The number of episodes saved in every
            file. Must set exactly one of `rollout_save_n_timesteps` and
            `rollout_save_n_episodes`.
        policy_save_interval: The number of training updates between in between
            intermediate rollout saves. If the argument is nonpositive, then
            don't save intermediate updates.
        policy_save_final: If True, then save the policy right after training is
            finished.
        agent_path: Path to load warm-started agent.
        _rnd: Random number generator provided by Sacred.

    Returns:
        The return value of `rollout_stats()` using the final policy.
    """
    sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()
    policy_dir = log_dir / "policies"

    policy_dir.mkdir(parents=True, exist_ok=True)

    expert_trajs = demonstrations.get_expert_trajectories()

    post_wrappers = [lambda env, idx: wrappers.RolloutInfoWrapper(env)]
    render_mode = "rgb_array"
    with environment.make_venv(  # type: ignore[wrong-keyword-args]
        post_wrappers=post_wrappers,
        env_make_kwargs=dict(render_mode=render_mode),
    ) as venv:
        # Train reward model.
        reward_net = reward.make_reward_net(venv)
        diffusion_trainer: DiffusionTrainer = diffusion.make_diffusion_trainer(
            demonstrations=expert_trajs,
            reward_net=reward_net,
            custom_logger=custom_logger,
        )

        def callback(round_num: int, /) -> None:
            if diffusion_save_interval > 0 and round_num % diffusion_save_interval == 0:
                save(
                    diffusion_trainer,
                    log_dir / "diffusions" / f"{round_num:06d}",
                )

        diffusion_loss = diffusion_trainer.train(
            diffusion_total_timesteps, callback=callback, log=False
        )

        logging.info(f"Final diffusion loss: {diffusion_loss}")
        if wandb.run is not None:
            wandb.run.summary["diffusion_loss"] = diffusion_loss

        if diffusion_save_final:
            save(diffusion_trainer, log_dir / "diffusions" / "final")

        # Train policy.
        callback_objs = []

        reward_net = diffusion_trainer.reward_test.to("cuda")
        reward_fn = reward_net.predict_processed
        logging.info(f"Loaded reward net with class {reward_net.__class__.__name__}")
        venv = RewardVecEnvWrapper(venv, reward_fn)
        callback_objs.append(venv.make_log_callback())

        if normalize_reward:
            # Normalize reward. Reward scale effectively changes the learning rate,
            # so normalizing it makes training more stable. Note we do *not* normalize
            # observations here; use the `NormalizeFeaturesExtractor` instead.
            venv = VecNormalize(venv, norm_obs=False, **normalize_kwargs)

        if policy_save_interval > 0:
            save_policy_callback: callbacks.EventCallback = (
                policies_serialize.SavePolicyCallback(policy_dir)
            )
            save_policy_callback = callbacks.EveryNTimesteps(
                policy_save_interval,
                save_policy_callback,
            )
            callback_objs.append(save_policy_callback)
        callback = callbacks.CallbackList(callback_objs)

        if agent_path is None:
            rl_algo = rl.make_rl_algo(venv)
        else:
            rl_algo = rl.load_rl_algo_from_path(agent_path=agent_path, venv=venv)
        rl_algo.set_logger(custom_logger)
        rl_algo.learn(total_timesteps, callback=callback, progress_bar=True)

        if policy_save_final:
            output_dir = policy_dir / "final"
            policies_serialize.save_stable_model(output_dir, rl_algo)

        # Final evaluation of expert policy.
        eval_stats = policy_evaluation.eval_policy(rl_algo, venv)
        return eval_stats


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_difo_na"
    observer = FileStorageObserver(observer_path)
    train_difo_na_ex.observers.append(observer)
    train_difo_na_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
