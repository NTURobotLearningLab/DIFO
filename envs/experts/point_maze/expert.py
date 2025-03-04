from typing import Dict, Optional, Tuple, Union

import numpy as np

from .controller import WaypointController


class PointMazeExpert:
    def __init__(
        self,
        venv,
        action_noise=0.0,
    ) -> None:
        self.action_noise = action_noise

        mazes = venv.get_attr("maze")
        self.waypoint_controllers = [WaypointController(maze) for maze in mazes]

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
