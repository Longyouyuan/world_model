"""State-Anchored TD-MPC2 agent."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from state_anchored.config import apply_state_anchored_defaults
from state_anchored.world_model import StateAnchoredWorldModel
from tdmpc2 import TDMPC2


class StateAnchoredTDMPC2(TDMPC2):
	"""TD-MPC2 agent with a raw-state recursive world-model carrier."""

	def __init__(self, cfg):
		# Do not call TDMPC2.__init__: it constructs the official latent model.
		torch.nn.Module.__init__(self)
		self.cfg = apply_state_anchored_defaults(cfg)
		self.device = torch.device("cuda:0")
		self.model = StateAnchoredWorldModel(self.cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{
				"params": self.model._encoder.parameters(),
				"lr": self.cfg.lr * self.cfg.sa_feature_lr_scale,
			},
			{"params": self.model._dynamics.parameters()},
			{"params": self.model._reward.parameters()},
			{
				"params": (
					self.model._termination.parameters()
					if self.cfg.episodic
					else []
				),
			},
			{"params": self.model._Qs.parameters()},
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(),
			lr=self.cfg.lr,
			eps=1e-5,
			capturable=True,
		)
		self.model.eval()
		self.scale = RunningScale(self.cfg)
		self.cfg.iterations += 2 * int(self.cfg.action_dim >= 20)
		self.discount = self._get_discount(self.cfg.episode_length)
		print("Episode length:", self.cfg.episode_length)
		print("Discount factor:", self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(
			self.cfg.horizon,
			self.cfg.action_dim,
			device=self.device,
		))
		self.register_buffer(
			"_sa_update_count",
			torch.tensor(0, dtype=torch.long, device=self.device),
		)
		if self.cfg.compile:
			print("Compiling update function with torch.compile...")
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@staticmethod
	def _module_grad_norm(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
		grad_norms = [
			parameter.grad.detach().norm(2)
			for parameter in module.parameters()
			if parameter.grad is not None
		]
		if not grad_norms:
			return torch.zeros((), device=device)
		return torch.stack(grad_norms).norm(2)

	@torch.no_grad()
	def _state_diagnostics(
		self,
		states: torch.Tensor,
		real_states: torch.Tensor,
	) -> dict[str, torch.Tensor]:
		std = self.model.state_norm.std.to(
			device=states.device, dtype=states.dtype
		)
		normalized_states = self.model.normalize_state(states)
		predicted_delta = (states[1:] - states[:-1]) / std
		diagnostics = {
			"state_norm_mean_abs": self.model.state_norm.mean.abs().mean(),
			"state_norm_std_mean": self.model.state_norm.std.mean(),
			"state_norm_std_min": self.model.state_norm.std.min(),
			"state_norm_std_max": self.model.state_norm.std.max(),
			"imagined_state_abs_max": states.abs().max(),
			"normalized_imagined_state_abs_max": normalized_states.abs().max(),
			"predicted_normalized_delta_abs_mean": predicted_delta.abs().mean(),
			"predicted_normalized_delta_abs_max": predicted_delta.abs().max(),
		}
		for step in (1, 3):
			if step < states.shape[0] and step < real_states.shape[0]:
				difference = (
					self.model.normalize_state(states[step])
					- self.model.normalize_state(real_states[step])
				)
				diagnostics[f"state_nrmse_{step}"] = (
					difference.square().mean(dim=-1).sqrt().mean()
				)
		return diagnostics

	def update(self, buffer):
		"""Update running state statistics, then run one compiled model update."""
		obs, action, reward, terminated, task = buffer.sample()
		if task is not None:
			raise NotImplementedError(
				"State-Anchored TD-MPC2 does not support multitask replay batches."
			)
		if self._sa_update_count.item() < self.cfg.sa_norm_freeze_updates:
			# Only actual replay observations are allowed to update these statistics.
			self.model.state_norm.update(obs)
		self._sa_update_count.add_(1)

		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated)

	def _update(self, obs, action, reward, terminated, task=None):
		self.model._assert_single_task(task)
		with torch.no_grad():
			next_state = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_state, reward, terminated, task)

		self.model.train()
		state_dim = self.cfg.obs_shape["state"][0]
		states = torch.empty(
			self.cfg.horizon + 1,
			self.cfg.batch_size,
			state_dim,
			device=self.device,
			dtype=obs.dtype,
		)
		state = self.model.encode(obs[0], task)
		states[0] = state
		state_loss = torch.zeros((), device=self.device, dtype=obs.dtype)
		for t, (_action, target_next_state) in enumerate(
			zip(action.unbind(0), next_state.unbind(0))
		):
			state = self.model.next(state, _action, task)
			predicted_normalized = self.model.normalize_state(state)
			target_normalized = self.model.normalize_state(target_next_state)
			if self.cfg.sa_state_loss == "mse":
				step_loss = F.mse_loss(predicted_normalized, target_normalized)
			else:
				step_loss = F.smooth_l1_loss(
					predicted_normalized,
					target_normalized,
					beta=self.cfg.sa_state_loss_beta,
				)
			state_loss = state_loss + step_loss * self.cfg.rho ** t
			states[t + 1] = state

		model_states = states[:-1]
		qs = self.model.Q(model_states, action, task, return_type="all")
		reward_preds = self.model.reward(model_states, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(
				states[1:], task, unnormalized=True
			)

		reward_loss = torch.zeros((), device=self.device, dtype=obs.dtype)
		value_loss = torch.zeros((), device=self.device, dtype=obs.dtype)
		for t, (
			reward_pred_t,
			reward_t,
			td_target_t,
			qs_t,
		) in enumerate(zip(
			reward_preds.unbind(0),
			reward.unbind(0),
			td_targets.unbind(0),
			qs.unbind(1),
		)):
			reward_loss = reward_loss + (
				math.soft_ce(reward_pred_t, reward_t, self.cfg).mean()
				* self.cfg.rho ** t
			)
			for q_t in qs_t.unbind(0):
				value_loss = value_loss + (
					math.soft_ce(q_t, td_target_t, self.cfg).mean()
					* self.cfg.rho ** t
				)

		state_loss = state_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(
				termination_pred, terminated
			)
		else:
			termination_loss = torch.zeros((), device=self.device, dtype=obs.dtype)
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.sa_state_coef * state_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.termination_coef * termination_loss
			+ self.cfg.value_coef * value_loss
		)

		diagnostics = (
			self._state_diagnostics(states.detach(), obs)
			if self.cfg.sa_log_diagnostics
			else {}
		)
		total_loss.backward()
		if self.cfg.sa_log_diagnostics:
			diagnostics.update({
				"feature_encoder_grad_norm": self._module_grad_norm(
					self.model._encoder, self.device
				),
				"dynamics_grad_norm": self._module_grad_norm(
					self.model._dynamics, self.device
				),
			})
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm
		)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		pi_info = self.update_pi(states.detach(), task)
		# Recomputed features create encoder gradients during the inherited actor
		# backward pass. The encoder is not an actor parameter; clear those grads
		# so they cannot leak into the next model update.
		self.optim.zero_grad(set_to_none=True)
		self.model.soft_update_target_Q()

		self.model.eval()
		info = {
			"consistency_loss": state_loss,
			"state_loss": state_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		}
		if self.cfg.episodic:
			info.update(math.termination_statistics(
				torch.sigmoid(termination_pred[-1]), terminated[-1]
			))
		info.update(pi_info)
		info.update(diagnostics)
		return {
			key: value.detach().mean()
			if isinstance(value, torch.Tensor)
			else torch.tensor(value, device=self.device)
			for key, value in info.items()
		}
