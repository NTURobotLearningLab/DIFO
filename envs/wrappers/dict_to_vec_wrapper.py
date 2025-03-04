from typing import List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.type_aliases import GymResetReturn, GymStepReturn


class DictToVecWrapper(gym.Wrapper):
    """
    Wrap a Dict observation space environment to a Vec observation space environment.

    :param env: the environment
    :param keys: the keys to concatenate
    """

    def __init__(
        self,
        env: gym.Env,
        keys: List[str] = ["observation", "desired_goal", "achieved_goal"],
    ):
        super().__init__(env)
        self._keys = keys

        # Check if the observation space is a Dict space
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), "Observation space must be a Dict space"

        # Check if the keys are in the observation space
        for key in keys:
            assert (
                key in env.observation_space.spaces
            ), f"Key {key} not found in observation space"

        # Create the Vec observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                np.sum([env.observation_space.spaces[key].shape[0] for key in keys]),
            ),
            dtype=env.observation_space.spaces[keys[0]].dtype,
        )

    def reset(self, **kwargs) -> GymResetReturn:
        """
        Reset the environment

        :return: observation
        """
        dict_obs, info = self.env.reset(**kwargs)

        vec_obs = [dict_obs[key] for key in self._keys]
        vec_obs = np.concatenate(vec_obs)

        if "success" in info:
            info["is_success"] = info.pop("success")

        return vec_obs, info

    def step(self, action) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        dict_obs, reward, terminated, truncated, info = self.env.step(action)

        vec_obs = [dict_obs[key] for key in self._keys]
        vec_obs = np.concatenate(vec_obs)

        if "success" in info:
            info["is_success"] = info.pop("success")

        return vec_obs, reward, terminated, truncated, info
