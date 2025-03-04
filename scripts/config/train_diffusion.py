import pathlib
import re
from glob import glob

import sacred
from sacred import SETTINGS

from scripts.ingredients import (
    demonstrations as demos_common,
    environment,
    logging as logging_ingredient,
    reward,
)

SETTINGS["CAPTURE_MODE"] = "sys"


train_diffusion_ex = sacred.Experiment(
    "train_diffusion",
    ingredients=[
        logging_ingredient.logging_ingredient,
        demos_common.demonstrations_ingredient,
        environment.environment_ingredient,
        reward.reward_ingredient,
    ],
)


@train_diffusion_ex.config
def config():
    show_config = True

    total_timesteps = int(1e6)  # Num of environment transitions to sample
    checkpoint_interval = 5000  # Num epochs between checkpoints (<0 disables)

    demo_batch_size = 64  # Batch size for training on demonstrations
    log_interval = 100


hyperparam_dir = pathlib.Path(__file__).absolute().parent / "tuned_hps"
config_files = glob(str(hyperparam_dir / "**/*.yaml"))

for config_file in config_files:
    config_name = re.search(r"([^/]+).yaml", config_file).group(1)
    train_diffusion_ex.add_named_config(config_name, str(config_file))
