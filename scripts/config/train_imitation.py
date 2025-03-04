"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import pathlib
import re
from glob import glob

import sacred
from sacred import SETTINGS

from scripts.ingredients import (
    bc,
    demonstrations as demos_common,
    environment,
    expert,
    iq_learn,
    logging as logging_ingredient,
    ot,
    policy_evaluation,
)

SETTINGS["CAPTURE_MODE"] = "sys"


train_imitation_ex = sacred.Experiment(
    "train_imitation",
    ingredients=[
        logging_ingredient.logging_ingredient,
        demos_common.demonstrations_ingredient,
        expert.expert_ingredient,
        environment.environment_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
        bc.bc_ingredient,
        iq_learn.iq_ingredient,
        ot.ot_ingredient,
    ],
)


@train_imitation_ex.config
def config():
    dagger = dict(
        use_offline_rollouts=False,  # warm-start policy with BC from offline demos
        total_timesteps=1e5,
        beta_schedule=None,
    )
    total_timesteps = int(1e6)  # Num of environment transitions to sample


@train_imitation_ex.named_config
def mountain_car():
    environment = dict(gym_id="MountainCar-v0")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_mountain_car():
    environment = dict(gym_id="seals/MountainCar-v0")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_cartpole():
    environment = dict(gym_id="seals/CartPole-v0")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def pendulum():
    environment = dict(gym_id="Pendulum-v1")


@train_imitation_ex.named_config
def ant():
    environment = dict(gym_id="Ant-v2")


@train_imitation_ex.named_config
def half_cheetah():
    environment = dict(gym_id="HalfCheetah-v2")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=60000)


@train_imitation_ex.named_config
def humanoid():
    environment = dict(gym_id="Humanoid-v2")


@train_imitation_ex.named_config
def seals_humanoid():
    environment = dict(gym_id="seals/Humanoid-v0")


@train_imitation_ex.named_config
def fast():
    dagger = dict(total_timesteps=50)
    bc = dict(train_kwargs=dict(n_batches=50))
    sqil = dict(total_timesteps=50)


hyperparam_dir = pathlib.Path(__file__).absolute().parent / "tuned_hps"
config_files = glob(str(hyperparam_dir / "**/*.yaml"))

for config_file in config_files:
    config_name = re.search(r"([^/]+).yaml", config_file).group(1)
    train_imitation_ex.add_named_config(config_name, str(config_file))
