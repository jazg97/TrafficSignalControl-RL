"""Modular Rainbow DQN network components for SUMO traffic-signal control.

The original Rainbow implementation in this repository lives inside a notebook
with a hard-coded architecture. This module provides the same core ideas in a
reusable form so hyperparameter optimization can search over architecture
choices, not only training hyperparameters.

Design goals:

- keep the input contract identical to the PPO state encoder: ``[3, 48, 46]``
- preserve the recurrent formulation used by the notebook implementation
- expose convolutional, recurrent, and dueling-head sizes as constructor args
- preserve Rainbow-specific components: noisy layers and categorical outputs
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RainbowNetworkConfig:
    """Architecture parameters intended to be tuned by search code later.

    The config mirrors the hard-coded notebook architecture but keeps each
    dimension explicit so experiments can record and reproduce exact model
    definitions.
    """

    input_channels: int = 3
    input_height: int = 48
    input_width: int = 46
    num_conv_layers: int = 2
    num_filters: tuple[int, ...] = (16, 32)
    kernel_sizes: tuple[int, ...] = (5, 3)
    pool_strides: tuple[int, ...] = (2, 2)
    lstm_units: int = 256
    shared_hidden_dim: int = 128
    advantage_hidden_dim: int = 128
    value_hidden_dim: int = 128
    noisy_std_init: float = 0.5

    def validate(self) -> None:
        """Fail early if the architecture definition is inconsistent."""
        if self.num_conv_layers < 1:
            raise ValueError("num_conv_layers must be at least 1")
        if len(self.num_filters) != self.num_conv_layers:
            raise ValueError("num_filters length must match num_conv_layers")
        if len(self.kernel_sizes) != self.num_conv_layers:
            raise ValueError("kernel_sizes length must match num_conv_layers")
        if len(self.pool_strides) != self.num_conv_layers:
            raise ValueError("pool_strides length must match num_conv_layers")


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer used by Rainbow DQN."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters following the NoisyNet paper."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Sample fresh factorized noise for the next forward passes."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the noisy affine transformation."""
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise with the standard transform."""
        noise = torch.randn(size)
        return noise.sign().mul(noise.abs().sqrt())


class ModularRainbowNetwork(nn.Module):
    """CNN+LSTM dueling categorical Q-network with configurable width/depth.

    The output distribution shape is ``[batch, action_dim, atom_size]``.
    The recurrent state follows PyTorch's LSTM convention:
    ``(h, c)`` with shape ``[1, batch, lstm_units]``.
    """

    def __init__(
        self,
        action_dim: int,
        atom_size: int,
        support: torch.Tensor,
        config: RainbowNetworkConfig,
    ):
        super().__init__()
        config.validate()

        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support = support
        self.config = config

        spatial_size = np.array((config.input_height, config.input_width))
        conv_blocks: list[nn.Module] = []
        in_channels = config.input_channels

        for out_channels, kernel_size, pool_stride in zip(
            config.num_filters, config.kernel_sizes, config.pool_strides
        ):
            conv_blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    padding="same",
                )
            )
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d((2, 2), stride=pool_stride))
            spatial_size = self.compute_output_size(
                spatial_size, kernel_size=2, stride=pool_stride
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_blocks)
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.conv_out_features = int(config.num_filters[-1] * np.prod(spatial_size))

        self.lstm = nn.LSTM(self.conv_out_features, config.lstm_units, 1, batch_first=True)
        self.shared_projection = nn.Linear(config.lstm_units, config.shared_hidden_dim)

        self.advantage_hidden = NoisyLinear(
            config.shared_hidden_dim, config.advantage_hidden_dim, config.noisy_std_init
        )
        self.advantage_out = NoisyLinear(
            config.advantage_hidden_dim, action_dim * atom_size, config.noisy_std_init
        )

        self.value_hidden = NoisyLinear(
            config.shared_hidden_dim, config.value_hidden_dim, config.noisy_std_init
        )
        self.value_out = NoisyLinear(
            config.value_hidden_dim, atom_size, config.noisy_std_init
        )

    @staticmethod
    def compute_output_size(
        input_size: np.ndarray,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> np.ndarray:
        """Compute the 2D output size after pooling for height and width."""
        return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def forward(self, x: torch.Tensor, h_in: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return expected Q-values and the next recurrent state."""
        dist, h_out = self.dist(x, h_in)
        q_values = torch.sum(dist * self.support, dim=2)
        return q_values, h_out

    def dist(self, x: torch.Tensor, h_in: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the categorical return distribution over atoms."""
        features = self.feature_extractor(x)
        features = features.view(-1, 1, self.conv_out_features)
        features, h_out = self.lstm(features, h_in)
        features = features.squeeze(1)
        features = torch.tanh(self.shared_projection(features))

        advantage = torch.tanh(self.advantage_hidden(features))
        advantage = self.advantage_out(advantage).view(-1, self.action_dim, self.atom_size)

        value = torch.tanh(self.value_hidden(features))
        value = self.value_out(value).view(-1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        # Keep support probabilities numerically safe while preserving a valid
        # categorical distribution.
        dist = dist.clamp(min=1e-3)
        dist = dist / dist.sum(dim=-1, keepdim=True)
        return dist, h_out

    def reset_noise(self) -> None:
        """Refresh noise in all dueling noisy layers."""
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()

    def initial_hidden(
        self, batch_size: int = 1, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a zeroed initial LSTM state with the configured width."""
        if device is None:
            device = self.support.device
        shape = (1, batch_size, self.config.lstm_units)
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device),
        )
