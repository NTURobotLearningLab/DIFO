"""Constructs deep network reward models."""

import collections
from typing import Any, Dict, Iterable, Optional, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.rewards.reward_nets import RewardNet, cnn_transpose
from imitation.util import networks


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


def build_cnn(
    observation_space: spaces.Space,
    in_channels: int,
    hid_channels: Iterable[int],
    out_size: int = 1,
    name: Optional[str] = None,
    activation: Type[nn.Module] = nn.ReLU,
    kernel_sizes: Union[int, Iterable[int]] = 3,
    strides: Union[int, Iterable[int]] = 1,
    paddings: Union[int, Iterable[int]] = 0,
    dropout_prob: float = 0.0,
    squeeze_output: bool = False,
    normalize_input_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    """Constructs a Torch CNN.

    Args:
        in_channels: number of channels of individual inputs; input to the CNN will have
            shape (batch_size, in_size, in_height, in_width).
        hid_channels: number of channels of hidden layers. If this is an empty iterable,
            then we build a linear function approximator.
        out_size: size of output vector.
        name: Name to use as a prefix for the layers ID.
        activation: activation to apply after hidden layers.
        kernel_size: size of convolutional kernels.
        stride: stride of convolutional kernels.
        padding: padding of convolutional kernels.
        dropout_prob: Dropout probability to use after each hidden layer. If 0,
            no dropout layers are added to the network.
        squeeze_output: if out_size=1, then squeeze_input=True ensures that CNN
            output is of size (B,) instead of (B,1).

    Returns:
        nn.Module: a CNN mapping from inputs of size (batch_size, in_size, in_height,
            in_width) to (batch_size, out_size), unless out_size=1 and
            squeeze_output=True, in which case the output is of size (batch_size, ).

    Raises:
        ValueError: if squeeze_output was supplied with out_size!=1.
    """
    layers: Dict[str, nn.Module] = {}

    if name is None:
        prefix = ""
    else:
        prefix = f"{name}_"

    image_size = observation_space.shape[1]
    input_shape = (1, in_channels, image_size, image_size)

    # Normalize input layer
    if normalize_input_layer:
        try:
            layer_instance = normalize_input_layer(input_shape)
        except TypeError as exc:
            raise ValueError(
                f"normalize_input_layer={normalize_input_layer} is not a valid "
                "normalization layer type accepting only one argument (in_size).",
            ) from exc
        layers[f"{prefix}normalize_input"] = layer_instance

    prev_channels = in_channels
    if isinstance(strides, int):
        strides = [strides] * len(hid_channels)
    if isinstance(paddings, int):
        paddings = [paddings] * len(hid_channels)
    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * len(hid_channels)
    for i, (n_channels, stride, padding, kernel_size) in enumerate(
        zip(hid_channels, strides, paddings, kernel_sizes),
    ):
        layers[f"{prefix}conv{i}"] = nn.Conv2d(
            prev_channels,
            n_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        prev_channels = n_channels
        if activation:
            layers[f"{prefix}act{i}"] = activation()
        if dropout_prob > 0.0:
            layers[f"{prefix}dropout{i}"] = nn.Dropout(dropout_prob)

    layers[f"{prefix}flatten"] = nn.Flatten()
    cnn = nn.Sequential(collections.OrderedDict(layers))
    # Compute shape by doing one forward pass
    with th.no_grad():
        input_tensor = th.zeros(input_shape)
        n_flatten = cnn(input_tensor).shape[1]

    # final dense layer
    layers[f"{prefix}dense_final"] = nn.Linear(n_flatten, out_size)

    if squeeze_output:
        if out_size != 1:
            raise ValueError("squeeze_output is only applicable when out_size=1")
        layers[f"{prefix}squeeze"] = SqueezeLayer()

    model = nn.Sequential(collections.OrderedDict(layers))
    return model


class CnnRewardNet(RewardNet):
    """CNN that takes as input the state, action, next state and done flag.

    Inputs are boosted to tensors with channel, height, and width dimensions, and then
    concatenated. Image inputs are assumed to be in (h,w,c) format, unless the argument
    hwc_format=False is passed in. Each input can be enabled or disabled by the `use_*`
    constructor keyword arguments, but either `use_state` or `use_next_state` must be
    True.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        cnn_kwargs: Dict[str, Any] = {},
        mlp_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Builds reward CNN.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: Should the current state be included as an input to the CNN?
            use_action: Should the current action be included as an input to the CNN?
            use_next_state: Should the next state be included as an input to the CNN?
            use_done: Should the "done" flag be included as an input to the CNN?
            hwc_format: Are image inputs in (h,w,c) format (True), or (c,h,w) (False)?
                If hwc_format is False, image inputs are not transposed.
            kwargs: Passed straight through to `build_cnn`.

        Raises:
            ValueError: if observation or action space is not easily massaged into a
                CNN input.
        """
        super().__init__(observation_space, action_space)
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.hwc_format = not preprocessing.is_image_space_channels_first(
            observation_space
        )

        if not (self.use_state or self.use_next_state):
            raise ValueError("CnnRewardNet must take current or next state as input.")

        if not preprocessing.is_image_space(observation_space):
            raise ValueError(
                "CnnRewardNet requires observations to be images.",
            )
        assert isinstance(observation_space, spaces.Box)  # Note: hint to mypy

        input_size = 0
        cnn_feature_size = 512
        mlp_input_size = cnn_feature_size

        if self.use_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_action:
            mlp_input_size += action_space.shape[0]

        if self.use_next_state:
            input_size += self.get_num_channels_obs(observation_space)

        if self.use_done:
            mlp_input_size += 1

        full_build_cnn_kwargs: Dict[str, Any] = {
            "observation_space": observation_space,
            "hid_channels": (32, 64, 64),
            "kernel_sizes": (8, 4, 3),
            "strides": (4, 2, 1),
            "paddings": 0,
            **cnn_kwargs,
            # we do not want the values below to be overridden
            "in_channels": input_size,
            "out_size": cnn_feature_size,
            "squeeze_output": cnn_feature_size == 1,
        }

        self.cnn = build_cnn(**full_build_cnn_kwargs)

        full_build_mlp_kwargs: Dict[str, Any] = {
            "hid_sizes": (32, 32),
            **mlp_kwargs,
            # we do not want the values below to be overridden
            "in_size": mlp_input_size,
            "out_size": 1,
            "squeeze_output": True,
        }

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)

    def get_num_channels_obs(self, space: spaces.Box) -> int:
        """Gets number of channels for the observation."""
        return space.shape[-1] if self.hwc_format else space.shape[0]

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Computes rewardNet value on input state, action, next_state, and done flag.

        Takes inputs that will be used, transposes image states to (c,h,w) format if
        needed, reshapes inputs to have compatible dimensions, concatenates them, and
        inputs them into the CNN.

        Args:
            state: current state.
            action: current action.
            next_state: next state.
            done: flag for whether the episode is over.

        Returns:
            th.Tensor: reward of the transition.
        """
        inputs = []
        if self.use_state:
            state_ = cnn_transpose(state) if self.hwc_format else state
            inputs.append(state_)
        if self.use_next_state:
            next_state_ = cnn_transpose(next_state) if self.hwc_format else next_state
            inputs.append(next_state_)

        inputs_concat = th.cat(inputs, dim=1)
        cnn_output = self.cnn(inputs_concat)

        mlp_inputs = [cnn_output]

        if self.use_action:
            mlp_inputs.append(action)
        if self.use_done:
            mlp_inputs.append(done)

        mlp_input = th.cat(mlp_inputs, dim=1)
        rewards = self.mlp(mlp_input)

        assert rewards.shape == state.shape[:1]

        return rewards
