from gymnasium.envs.registration import register

from . import wrappers
from .maze import maps


def _merge(a, b):
    a.update(b)
    return a


for reward_type in ["sparse", "dense"]:
    reward_suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    # AntMaze
    register(
        id=f"AntMazeRandPointOne_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.1},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointTwo_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.2},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointThree_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.3},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointFour_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.4},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointFive_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.5},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointSix_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.6},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointSeven_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.7},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointEight_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.8},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandPointNine_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 0.9},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeRandLarge_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {"maze_map": maps.U_MAZE, "action_noise": 1.0},
            kwargs,
        ),
        max_episode_steps=700,
    )
    register(
        id=f"AntMazeCustom_Medium_Diverse_GR{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {
                "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
            },
            kwargs,
        ),
        max_episode_steps=1000,
    )

    register(
        id=f"AntMazeCustom_UMaze{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {
                "maze_map": maps.U_MAZE,
            },
            kwargs,
        ),
        max_episode_steps=700,
    )

    register(
        id=f"AntMaze_Cross{reward_suffix}-v4",
        entry_point="envs.maze.ant_maze_v4:AntMazeEnv",
        kwargs=_merge(
            {
                "maze_map": maps.EDGES_G,
            },
            kwargs,
        ),
        max_episode_steps=100,
    )

    # Maze
    register(
        id=f"PointMazeCustom_Medium{reward_suffix}-v3",
        entry_point="envs.maze.point_maze:PointMazeCustomEnv",
        kwargs=_merge(
            {
                "maze_map": maps.MEDIUM_MAZE,
            },
            kwargs,
        ),
        max_episode_steps=600,
    )

    register(
        id=f"PointMazeShort_Medium{reward_suffix}_GR-v3",
        entry_point="envs.maze.point_maze:PointMazeCustomEnv",
        kwargs=_merge(
            {
                "maze_map": maps.MEDIUM_MAZE_DIVERSE_GR,
                "continuing_task": False,
            },
            kwargs,
        ),
        max_episode_steps=600,
    )

    register(
        id=f"PointMazeDone_Medium{reward_suffix}-v3",
        entry_point="envs.maze.point_maze:PointMazeCustomEnv",
        kwargs=_merge(
            {
                "maze_map": maps.MEDIUM_MAZE,
                "continuing_task": False,
            },
            kwargs,
        ),
        max_episode_steps=600,
    )
    register(
        id=f"PointMazeRand_Medium{reward_suffix}-v3",
        entry_point="envs.maze.point_maze:PointMazeCustomEnv",
        kwargs=_merge(
            {
                "maze_map": maps.MEDIUM_MAZE,
                "continuing_task": False,
                "action_noise": 0.2,
            },
            kwargs,
        ),
        max_episode_steps=600,
    )

    # Fetch
    register(
        id=f"FetchPushCustom{reward_suffix}-v2",
        entry_point="envs.fetch.push:MujocoFetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id=f"FetchPushStatic{reward_suffix}-v2",
        entry_point="envs.fetch.push:MujocoFetchPushEnv",
        kwargs=_merge(
            {
                "use_velocity_obs": False,
            },
            kwargs,
        ),
        max_episode_steps=50,
    )

    # Hand
    register(
        id=f"AdroitHandDoorCustom{reward_suffix}-v1",
        entry_point="envs.adroit_hand.adroit_door:AdroitHandDoorEnv",
        max_episode_steps=200,
        kwargs=kwargs,
    )


register(
    id="Walker2d-fixed-v4",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv",
    kwargs=dict(
        terminate_when_unhealthy=False,
    ),
)

register(
    id="SineEnv-v0",
    max_episode_steps=1,
    entry_point="envs.sine.sine_env:SineEnv",
    kwargs=dict(
        noisy=False,
    ),
)

register(
    id="FrankaKitchenCustom-v1",
    entry_point="envs.franka_kitchen:KitchenEnv",
    max_episode_steps=60,
    kwargs=dict(
        remove_task_when_completed=True,
        terminate_on_tasks_completed=False,
        tasks_to_complete=["microwave"],
        object_noise_ratio=0.0,
    ),
)
