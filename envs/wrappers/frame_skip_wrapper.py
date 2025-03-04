import gymnasium as gym
from stable_baselines3.common.type_aliases import GymStepReturn


class FrameSkip(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
