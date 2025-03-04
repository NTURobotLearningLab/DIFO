from typing import Dict, Optional, Tuple, Union

import numpy as np
from stable_baselines3 import SAC

from .controller import WaypointController


def wrap_maze_obs(obs, waypoint_xy):
    """Wrap the maze obs into one suitable for GoalReachAnt."""
    goal_direction = waypoint_xy - obs["achieved_goal"]
    observation = np.concatenate([obs["observation"], goal_direction])
    return observation


class AntMazeExpert:
    def __init__(
        self,
        venv,
        action_noise=0.0,
        policy_path="/home/ray/d4rl-minari-dataset-generation/scripts/antmaze/GoalReachAnt_model.zip",
    ) -> None:
        self.model = SAC.load(policy_path)
        self.action_noise = action_noise

        mazes = venv.get_attr("maze")
        self.waypoint_controllers = [
            WaypointController(maze, self.action_callback) for maze in mazes
        ]

    def action_callback(self, obs, waypoint_xy):
        return self.model.predict(wrap_maze_obs(obs, waypoint_xy))[0]

    def __call__(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        done: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        actions = []
        for i in range(len(self.waypoint_controllers)):
            obs = {k: v[i] for k, v in observation.items()}
            action = self.waypoint_controllers[i].compute_action(obs)
            action += self.action_noise * np.random.randn(*action.shape)
            action = np.clip(action, -1, 1)
            actions.append(action)
        action = np.stack(actions)
        return action, None
