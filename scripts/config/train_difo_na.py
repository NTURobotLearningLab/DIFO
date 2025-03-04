"""Configuration settings for train_difo_na."""

import pathlib
import re
from glob import glob

import sacred
from sacred import SETTINGS

from scripts.ingredients import (
    demonstrations,
    diffusion,
    environment,
    logging as logging_ingredient,
    policy_evaluation,
    reward,
    rl,
)

SETTINGS["CAPTURE_MODE"] = "sys"


train_difo_na_ex = sacred.Experiment(
    "train_difo_na",
    ingredients=[
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
        rl.rl_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
        diffusion.diffusion_ingredient,
        demonstrations.demonstrations_ingredient,
        reward.reward_ingredient,
    ],
)


@train_difo_na_ex.config
def train_difo_na_defaults():
    diffusion_total_timesteps = int(1e5)  # Num of environment transitions to sample
    total_timesteps = int(1e6)  # Number of training timesteps in model.learn()
    normalize_reward = True  # Use VecNormalize to normalize the reward
    normalize_kwargs = dict()  # kwargs for `VecNormalize`

    policy_save_interval = 10000  # Num timesteps between saves (<=0 disables)
    policy_save_final = True  # If True, save after training is finished.

    diffusion_save_interval = 5000  # Num timesteps between saves (<=0 disables)
    diffusion_save_final = True  # If True, save after training is finished.

    agent_path = None  # Path to load agent from, optional.


# Debug configs


@train_difo_na_ex.named_config
def fast():
    # Intended for testing purposes: small # of updates, ends quickly.
    diffusion_total_timesteps = int(1000)
    total_timesteps = int(10000)
    policy_save_interval = 500
    diffusion_save_interval = 100


hyperparam_dir = pathlib.Path(__file__).absolute().parent / "tuned_hps"
config_files = glob(str(hyperparam_dir / "**/*.yaml"))

for config_file in config_files:
    config_name = re.search(r"([^/]+).yaml", config_file).group(1)
    train_difo_na_ex.add_named_config(config_name, str(config_file))
