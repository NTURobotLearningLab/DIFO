import os
import pickle
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from imitation.data import types


def min_max_norm(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y))


np.random.seed(0)


def generate_data():
    # sine
    f = 3
    x = np.linspace(0, 1, 500, endpoint=True)
    y = min_max_norm(np.sin(2 * np.pi * f * x) + x * 4)
    # collect data
    scale = 20
    s = np.repeat(x, scale)
    s_t_plus_1 = np.repeat(y, scale)
    noise = np.random.normal(0, 0.05, s_t_plus_1.shape)
    s_t_plus_1_noise = s_t_plus_1 + noise
    s_s_t_plus_1_noise = np.append(s, min_max_norm(s_t_plus_1_noise))
    s = s_s_t_plus_1_noise[: len(s)]
    s_t_plus_1_noise = s_s_t_plus_1_noise[len(s) :]
    a = s_t_plus_1_noise - s
    return s, a, s_t_plus_1_noise


def generate_trajectories() -> Sequence[types.TrajectoryWithRew]:
    trajectories = []
    s_t, a, s_t_plus_1 = generate_data()
    for i in range(len(s_t)):
        obs, acts, rews = (
            np.array([s_t[i], s_t_plus_1[i]]),
            np.array([a[i]]),
            np.array([0.0]),
        )
        trajectories.append(
            types.TrajectoryWithRew(
                obs=obs, acts=acts, rews=rews, terminal=False, infos=[{}]
            )
        )
    return trajectories


if __name__ == "__main__":
    rollout_save_path = Path(f"/tmp2/drail/expert_policies/sine/rollouts/expert.pkl")
    if not rollout_save_path.parent.exists():
        rollout_save_path.parent.mkdir(parents=True)

    trajectories = generate_trajectories()
    random.shuffle(trajectories)
    if rollout_save_path:
        print(f"Saving rollouts to {rollout_save_path}")
        if not os.path.exists(rollout_save_path.parent):
            os.makedirs(rollout_save_path.parent, exist_ok=True)
        pickle.dump(trajectories, open(rollout_save_path, "wb"))
        print(
            f"Size of rollout save path: {os.path.getsize(rollout_save_path) / 1e6:.2f} MB"
        )
    print()
