import copy
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, TypeVar

import gymnasium as gym

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class TerminalPixelInfoWrapper(gym.Wrapper):
    """Wrapper that adds pixels to info dict."""

    def __init__(
        self,
        env: gym.Env,
        render_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_keys: Tuple[str, ...] = ("pixels",),
    ):
        gym.Wrapper.__init__(self, env)

        # Avoid side-effects that occur when render_kwargs is manipulated
        render_kwargs = copy.deepcopy(render_kwargs)
        self.render_history = []

        if render_kwargs is None:
            render_kwargs = {}

        for key in render_kwargs:
            assert key in pixel_keys, (
                "The argument render_kwargs should map elements of "
                "pixel_keys to dictionaries of keyword arguments. "
                f"Found key '{key}' in render_kwargs but not in pixel_keys."
            )

        default_render_kwargs = {}
        if not env.render_mode:
            raise AttributeError(
                "env.render_mode must be specified to use PixelObservationWrapper:"
                "`gymnasium.make(env_name, render_mode='rgb_array')`."
            )

        for key in pixel_keys:
            render_kwargs.setdefault(key, default_render_kwargs)

        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`step` using :meth:`self.observation` on the returned observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = self._add_pixel_info(info)
        return observation, reward, terminated, truncated, info

    def _add_pixel_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        pixel_observations = {
            pixel_key: self._render(**self._render_kwargs[pixel_key])
            for pixel_key in self._pixel_keys
        }

        info.update(pixel_observations)

        return info

    def render(self, *args, **kwargs):
        """Renders the environment."""
        render = self.env.render(*args, **kwargs)
        if isinstance(render, list):
            render = self.render_history + render
            self.render_history = []
        return render

    def _render(self, *args, **kwargs):
        render = self.env.render(*args, **kwargs)
        if isinstance(render, list):
            self.render_history += render
        return render
