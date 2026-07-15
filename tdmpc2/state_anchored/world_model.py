"""World model whose recursive carrier is the unclipped normalized state."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from common import init, layers as common_layers, math
from state_anchored.config import apply_state_anchored_defaults
from state_anchored.layers import (
	RunningStateNorm,
	StateDeltaDynamics,
	StateFeatureEncoder,
)


class StateAnchoredWorldModel(nn.Module):
	"""TD-MPC2 world model that rolls out normalized states instead of latents."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = apply_state_anchored_defaults(cfg)
		assert self.cfg.obs == "state"
		assert not self.cfg.multitask

		self.state_dim = int(self.cfg.obs_shape["state"][0])
		self.feature_dim = int(self.cfg.sa_feature_dim)
		head_dim = self.state_dim + self.feature_dim

		self.state_norm = RunningStateNorm(
			self.state_dim,
			eps=self.cfg.sa_norm_eps,
			min_std=self.cfg.sa_norm_min_std,
			clip=self.cfg.sa_norm_clip,
		)
		# `_encoder` is an auxiliary feature encoder, not a carrier encoder.
		self._encoder = StateFeatureEncoder(self.state_dim, self.cfg)
		self._dynamics = StateDeltaDynamics(
			self.state_dim,
			self.feature_dim,
			self.cfg.action_dim,
			self.cfg,
		)
		self._reward = common_layers.mlp(
			head_dim + self.cfg.action_dim,
			2 * [self.cfg.mlp_dim],
			max(self.cfg.num_bins, 1),
		)
		self._termination = (
			common_layers.mlp(head_dim, 2 * [self.cfg.mlp_dim], 1)
			if self.cfg.episodic
			else None
		)
		self._pi = common_layers.mlp(
			head_dim,
			2 * [self.cfg.mlp_dim],
			2 * self.cfg.action_dim,
		)
		self._Qs = common_layers.Ensemble([
			common_layers.mlp(
				head_dim + self.cfg.action_dim,
				2 * [self.cfg.mlp_dim],
				max(self.cfg.num_bins, 1),
				dropout=self.cfg.dropout,
			).apply(init.weight_init)
			for _ in range(self.cfg.num_q)
		])

		self.apply(init.weight_init)
		init.zero_([
			self._reward[-1].weight,
			self._Qs.params["2", "weight"],
		])
		if self.cfg.sa_zero_init_dynamics_output:
			init.zero_([
				self._dynamics.output_layer.weight,
				self._dynamics.output_layer.bias,
			])

		self.register_buffer("log_std_min", torch.tensor(self.cfg.log_std_min))
		self.register_buffer(
			"log_std_dif",
			torch.tensor(self.cfg.log_std_max) - self.log_std_min,
		)
		self.init()

	def init(self) -> None:
		"""Create detached and target views of the vectorized Q ensemble."""
		self._detach_Qs_params = TensorDictParams(
			self._Qs.params.data, no_convert=True
		)
		self._target_Qs_params = TensorDictParams(
			self._Qs.params.data.clone(), no_convert=True
		)

		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params
		self._target_Qs.train(False)

	@property
	def total_params(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	@property
	def parameter_counts(self) -> dict[str, int]:
		"""Return learnable parameter counts for each model component."""
		def count(module: nn.Module | None) -> int:
			if module is None:
				return 0
			return sum(p.numel() for p in module.parameters() if p.requires_grad)

		counts = {
			"feature_encoder": count(self._encoder),
			"dynamics": count(self._dynamics),
			"reward": count(self._reward),
			"termination": count(self._termination),
			"policy": count(self._pi),
			"q_ensemble": count(self._Qs),
		}
		counts["total"] = sum(counts.values())
		return counts

	def __repr__(self) -> str:
		modules = [
			("State normalizer", self.state_norm),
			("Feature encoder", self._encoder),
			("State delta dynamics", self._dynamics),
			("Reward", self._reward),
			("Termination", self._termination),
			("Policy prior", self._pi),
			("Q-functions", self._Qs),
		]
		lines = ["State-Anchored TD-MPC2 World Model"]
		for name, module in modules:
			if module is None:
				continue
			lines.append(f"{name}: {module}")
		lines.extend([
			f"Carrier dimension: {self.state_dim}",
			f"Auxiliary feature dimension: {self.feature_dim}",
			f"Learnable parameters: {self.total_params:,}",
		])
		return "\n".join(lines)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode: bool = True):
		"""Keep target Q-functions in evaluation mode."""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self) -> None:
		"""Polyak-average the target Q-functions toward live Q-functions."""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	@staticmethod
	def _assert_single_task(task) -> None:
		assert task is None, "State-Anchored TD-MPC2 does not support task IDs."

	def encode(self, obs: torch.Tensor, task=None) -> torch.Tensor:
		"""Map a raw state to the unclipped normalized recursive carrier."""
		self._assert_single_task(task)
		return self.state_norm.normalize_unclipped(obs)

	def decode_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
		"""Map an unclipped normalized carrier back to raw state space."""
		return self.state_norm.denormalize(normalized_state)

	def features(self, normalized_state: torch.Tensor) -> torch.Tensor:
		"""Recompute anchored features from the current normalized carrier."""
		network_state = self.state_norm.clip_normalized(normalized_state)
		feature = self._encoder(network_state)
		return torch.cat([network_state, feature], dim=-1)

	def next(
		self,
		normalized_state: torch.Tensor,
		action: torch.Tensor,
		task=None,
	) -> torch.Tensor:
		"""Predict the next carrier using a bounded normalized residual."""
		self._assert_single_task(task)
		anchored = self.features(normalized_state)
		raw_delta = self._dynamics(torch.cat([anchored, action], dim=-1))
		limit = self.cfg.sa_delta_limit
		bounded_delta = limit * torch.tanh(raw_delta / limit)
		return normalized_state + bounded_delta

	def reward(self, state: torch.Tensor, action: torch.Tensor, task=None) -> torch.Tensor:
		self._assert_single_task(task)
		anchored = self.features(state)
		return self._reward(torch.cat([anchored, action], dim=-1))

	def termination(
		self,
		state: torch.Tensor,
		task=None,
		unnormalized: bool = False,
	) -> torch.Tensor:
		self._assert_single_task(task)
		if self._termination is None:
			raise RuntimeError("Termination head is disabled when cfg.episodic is false.")
		logits = self._termination(self.features(state))
		return logits if unnormalized else torch.sigmoid(logits)

	def pi(self, state: torch.Tensor, task=None):
		"""Sample an action from the official Gaussian policy prior."""
		self._assert_single_task(task)
		mean, log_std = self._pi(self.features(state)).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		log_prob = math.gaussian_logprob(eps, log_std)
		scaled_log_prob = log_prob * eps.shape[-1]
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.0,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(
		self,
		state: torch.Tensor,
		action: torch.Tensor,
		task=None,
		return_type: str = "min",
		target: bool = False,
		detach: bool = False,
	) -> torch.Tensor:
		"""Evaluate the official distributional Q ensemble on anchored features."""
		self._assert_single_task(task)
		if return_type not in {"min", "avg", "all"}:
			raise ValueError("return_type must be one of {'min', 'avg', 'all'}.")

		q_input = torch.cat([self.features(state), action], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(q_input)

		if return_type == "all":
			return out
		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		q_values = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return q_values.min(0).values
		return q_values.sum(0) / 2
