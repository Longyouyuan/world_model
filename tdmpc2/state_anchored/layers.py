"""State-anchored representation, normalization, and dynamics layers."""

from __future__ import annotations

import torch
import torch.nn as nn

from common import layers as common_layers


class RunningStateNorm(nn.Module):
	"""Running per-dimension state statistics using batch Welford updates."""

	def __init__(
		self,
		state_dim: int,
		eps: float = 1e-5,
		min_std: float = 1e-3,
		clip: float | None = 10.0,
	):
		super().__init__()
		if state_dim <= 0:
			raise ValueError("state_dim must be positive.")
		if eps < 0:
			raise ValueError("eps must be non-negative.")
		if min_std <= 0:
			raise ValueError("min_std must be positive.")
		if clip is not None and clip <= 0:
			raise ValueError("clip must be positive or None.")

		self.state_dim = int(state_dim)
		self.eps = float(eps)
		self.min_std = float(min_std)
		self.clip = None if clip is None else float(clip)
		self.register_buffer("count", torch.zeros((), dtype=torch.float64))
		self.register_buffer("mean", torch.zeros(self.state_dim, dtype=torch.float64))
		self.register_buffer("m2", torch.zeros(self.state_dim, dtype=torch.float64))

	def _check_shape(self, states: torch.Tensor) -> None:
		if states.ndim < 1 or states.shape[-1] != self.state_dim:
			raise ValueError(
				f"Expected states with final dimension {self.state_dim}, "
				f"got shape {tuple(states.shape)}."
			)

	@torch.no_grad()
	def update(self, states: torch.Tensor) -> None:
		"""Merge a batch of real states into the running statistics."""
		self._check_shape(states)
		batch = states.detach().reshape(-1, self.state_dim)
		if batch.shape[0] == 0:
			return
		batch = batch.to(device=self.mean.device, dtype=torch.float64)
		batch_count = torch.as_tensor(
			batch.shape[0], device=self.count.device, dtype=torch.float64
		)
		batch_mean = batch.mean(dim=0)
		batch_m2 = (batch - batch_mean).square().sum(dim=0)

		delta = batch_mean - self.mean
		total_count = self.count + batch_count
		new_mean = self.mean + delta * (batch_count / total_count)
		new_m2 = (
			self.m2
			+ batch_m2
			+ delta.square() * self.count * batch_count / total_count
		)

		self.mean.copy_(new_mean)
		self.m2.copy_(new_m2)
		self.count.copy_(total_count)

	@property
	def var(self) -> torch.Tensor:
		"""Return the unbiased running variance, or ones before two samples."""
		denominator = (self.count - 1).clamp_min(1)
		estimate = (self.m2 / denominator).clamp_min(0)
		return torch.where(self.count < 2, torch.ones_like(estimate), estimate)

	@property
	def std(self) -> torch.Tensor:
		"""Return a numerically stable standard deviation."""
		denominator = (self.count - 1).clamp_min(1)
		variance = (self.m2 / denominator).clamp_min(0)
		estimate = (variance + self.eps).sqrt().clamp_min(self.min_std)
		return torch.where(self.count < 2, torch.ones_like(estimate), estimate)

	def _stats_like(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		self._check_shape(tensor)
		mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
		std = self.std.to(device=tensor.device, dtype=tensor.dtype)
		return mean, std

	def normalize(self, state: torch.Tensor) -> torch.Tensor:
		"""Normalize raw state while preserving its dtype and device."""
		mean, std = self._stats_like(state)
		normalized = (state - mean) / std
		if self.clip is not None:
			normalized = normalized.clamp(-self.clip, self.clip)
		return normalized

	def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
		"""Map an unclipped normalized state back to raw state space."""
		mean, std = self._stats_like(normalized)
		return normalized * std + mean

	def scale_delta(self, normalized_delta: torch.Tensor) -> torch.Tensor:
		"""Map normalized deltas to raw deltas without clipping."""
		self._check_shape(normalized_delta)
		std = self.std.to(
			device=normalized_delta.device, dtype=normalized_delta.dtype
		)
		return normalized_delta * std

	def __repr__(self) -> str:
		return (
			f"RunningStateNorm(state_dim={self.state_dim}, eps={self.eps}, "
			f"min_std={self.min_std}, clip={self.clip}, count={self.count.item():g})"
		)


class StateFeatureEncoder(nn.Module):
	"""Encode normalized state into an auxiliary learned feature."""

	def __init__(self, state_dim: int, cfg):
		super().__init__()
		activation = (
			common_layers.SimNorm(cfg)
			if cfg.sa_feature_simnorm
			else nn.Identity()
		)
		self.net = common_layers.mlp(
			state_dim,
			max(cfg.num_enc_layers - 1, 1) * [cfg.sa_feature_hidden_dim],
			cfg.sa_feature_dim,
			act=activation,
		)

	def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
		return self.net(normalized_state)


class StateDeltaDynamics(nn.Module):
	"""Predict a normalized state delta from anchored features and action."""

	def __init__(self, state_dim: int, feature_dim: int, action_dim: int, cfg):
		super().__init__()
		self.net = common_layers.mlp(
			state_dim + feature_dim + action_dim,
			cfg.sa_dynamics_layers * [cfg.sa_dynamics_hidden_dim],
			state_dim,
		)

	def forward(self, anchored_state_action: torch.Tensor) -> torch.Tensor:
		return self.net(anchored_state_action)

	@property
	def output_layer(self) -> nn.Linear:
		layer = self.net[-1]
		if not isinstance(layer, nn.Linear):
			raise TypeError("State dynamics output layer must be nn.Linear.")
		return layer
