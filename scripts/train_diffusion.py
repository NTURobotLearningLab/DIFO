import pathlib
from typing import Mapping

import numpy as np
import sacred.commands
import torch as th
from sacred.observers import FileStorageObserver

from difo.difo import DiffusionTrainer
from scripts.config.train_diffusion import train_diffusion_ex
from scripts.ingredients import (
    demonstrations,
    environment,
    logging as logging_ingredient,
    reward,
)


def save(trainer: DiffusionTrainer, save_path: pathlib.Path):
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_test, save_path / "reward_test.pt")
    print("Saved reward_test to %s", save_path / "reward_test.pt")


@train_diffusion_ex.config_hook
def hook(config, command_name, logger):
    environment = config["environment"]
    environment.update(dict(num_vec=1, parallel=False))
    config.update(environment=environment)
    return config


@train_diffusion_ex.main
def train_diffusion(
    *,
    _run,
    show_config: bool,
    total_timesteps: int,
    demo_batch_size: int,
    checkpoint_interval: int,
    _rnd: np.random.Generator,
) -> Mapping[str, float]:
    # This allows to specify total_timesteps and checkpoint_interval in scientific
    # notation, which is interpreted as a float by python.
    total_timesteps = int(total_timesteps)
    checkpoint_interval = int(checkpoint_interval)

    if show_config:
        # Running `train_adversarial print_config` will show unmerged config.
        # So, support showing merged config from `train_adversarial {airl,gail}`.
        sacred.commands.print_config(_run)

    custom_logger, log_dir = logging_ingredient.setup_logging()
    expert_trajs = demonstrations.get_expert_trajectories()

    render_mode = "rgb_array"
    with environment.make_venv(  # type: ignore[wrong-keyword-args]
        env_make_kwargs=dict(render_mode=render_mode),
    ) as venv:
        reward_net = reward.make_reward_net(venv)

        trainer = DiffusionTrainer(
            demonstrations=expert_trajs,
            demo_batch_size=demo_batch_size,
            reward_net=reward_net,
            custom_logger=custom_logger,
        )

        def callback(round_num: int, /) -> None:
            if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
                save(trainer, log_dir / "checkpoints" / f"{round_num:06d}")

        trainer.train(total_timesteps, callback)

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save(trainer, log_dir / "checkpoints" / "final")


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_diffusion"
    observer = FileStorageObserver(observer_path)
    train_diffusion_ex.observers.append(observer)
    train_diffusion_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
