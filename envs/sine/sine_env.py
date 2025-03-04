import os
from typing import Dict, List, Optional, Union

import gymnasium.spaces as spaces
import numpy as np
from gymnasium.core import Env
from gymnasium.utils.ezpickle import EzPickle


class SineEnv(Env, EzPickle):
    """
    ## Description


    ## Action Space
    delta x

    ## Observation Space
    1D-space

    ## Rewards
    no reward
    """

    def __init__(self, noisy=False, render_mode=None, **kwargs):
        self.ACTION_RANGE = 0.45
        self.STATE_RANGE = (0, 1)
        self.reset_pos = np.array(
            [np.random.uniform(self.STATE_RANGE[0], self.STATE_RANGE[1])],
            dtype=np.float32,
        )
        self.state = self.reset_pos
        self.noisy = noisy
        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=-self.ACTION_RANGE, high=self.ACTION_RANGE, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.STATE_RANGE[0],
            high=self.STATE_RANGE[1],
            shape=(1,),
            dtype=np.float32,
        )
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        EzPickle.__init__(self, noisy=noisy, **kwargs)

    def reset(self, *, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        np.random.seed(seed)
        self.state = np.array(
            [np.random.uniform(self.STATE_RANGE[0], self.STATE_RANGE[1])],
            dtype=np.float32,
        )
        return self.state, {}

    def step(self, action):
        noise = 0
        if self.noisy:
            noise = np.random.normal(0, 0.01)
        action_clip = np.clip(action + noise, -self.ACTION_RANGE, self.ACTION_RANGE)
        self.state = np.clip(
            action_clip + self.state, self.STATE_RANGE[0], self.STATE_RANGE[1]
        )
        reward = 0
        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        super().close()

    @property
    def model(self):
        return None

    @property
    def data(self):
        return self.data
