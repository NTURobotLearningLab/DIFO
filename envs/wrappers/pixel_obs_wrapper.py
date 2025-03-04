"""Wrapper for augmenting observations by pixel values."""

import collections
import copy
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PixelObservationWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Augment observations by pixel values.

    Observations of this wrapper will be dictionaries of images.
    You can also choose to add the observation of the base environment to this dictionary.
    In that case, if the base environment has an observation space of type :class:`Dict`, the dictionary
    of rendered images will be updated with the base environment's observation. If, however, the observation
    space is of type :class:`Box`, the base environment's observation (which will be an element of the :class:`Box`
    space) will be added to the dictionary under the key "state".

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import PixelObservationWrapper
        >>> env = PixelObservationWrapper(gym.make("CarRacing-v2", render_mode="rgb_array"))
        >>> obs, _ = env.reset()
        >>> obs.keys()
        odict_keys(['pixels'])
        >>> obs['pixels'].shape
        (400, 600, 3)
        >>> env = PixelObservationWrapper(gym.make("CarRacing-v2", render_mode="rgb_array"), pixels_only=False)
        >>> obs, _ = env.reset()
        >>> obs.keys()
        odict_keys(['state', 'pixels'])
        >>> obs['state'].shape
        (96, 96, 3)
        >>> obs['pixels'].shape
        (400, 600, 3)
        >>> env = PixelObservationWrapper(gym.make("CarRacing-v2", render_mode="rgb_array"), pixel_keys=('obs',))
        >>> obs, _ = env.reset()
        >>> obs.keys()
        odict_keys(['obs'])
        >>> obs['obs'].shape
        (400, 600, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only (bool): If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs (dict): Optional dictionary containing that maps elements of `pixel_keys` to
                keyword arguments passed to the :meth:`self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to `(pixels,)`.

        Raises:
            AssertionError: If any of the keys in ``render_kwargs``do not show up in ``pixel_keys``.
            ValueError: If ``env``'s observation space is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If ``env``'s observation already contains any of the
                specified ``pixel_keys``.
            TypeError: When an unexpected pixel type is used
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            render_kwargs=render_kwargs,
        )
        gym.ObservationWrapper.__init__(self, env)

        # Avoid side-effects that occur when render_kwargs is manipulated
        render_kwargs = copy.deepcopy(render_kwargs)
        self.render_history = []

        if render_kwargs is None:
            render_kwargs = {}

        # Extend observation space with pixels.

        self.env.reset()

        pixels = self._render(**render_kwargs)
        pixels: np.ndarray = pixels[-1] if isinstance(pixels, List) else pixels

        if not hasattr(pixels, "dtype") or not hasattr(pixels, "shape"):
            raise TypeError(
                f"Render method returns a {pixels.__class__.__name__}, but an array with dtype and shape is expected."
                "Be sure to specify the correct render_mode."
            )

        if np.issubdtype(pixels.dtype, np.integer):
            low, high = (0, 255)
        elif np.issubdtype(pixels.dtype, np.float):
            low, high = (-float("inf"), float("inf"))
        else:
            raise TypeError(pixels.dtype)

        pixels_space = spaces.Box(
            shape=pixels.shape, low=low, high=high, dtype=pixels.dtype
        )

        self.observation_space = pixels_space

        self._render_kwargs = render_kwargs

    def observation(self, observation):
        """Updates the observations with the pixel observations.

        Args:
            observation: The observation to add pixel observations for

        Returns:
            The updated pixel observations
        """
        return self._render(**self._render_kwargs)

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
