"""State-Anchored TD-MPC2 agent."""

from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from state_anchored.config import apply_state_anchored_defaults
from state_anchored.world_model import StateAnchoredWorldModel
from tdmpc2 import TDMPC2


class StateAnchoredTDMPC2(TDMPC2):
	"""TD-MPC2 agent with an unclipped normalized-state model carrier."""

	_CHECKPOINT_VERSION = 2

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
		self._norm_initialized: bool = False
		self._sa_update_count_cpu: int = 0
		if self.cfg.compile:
			print("Compiling update function with torch.compile...")
			# self._update = torch.compile(self._update, mode="reduce-overhead")
			self._update = torch.compile(self._update, mode="default")

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
		raw_states = self.model.decode_state(states)
		real_x = self.model.encode(real_states)
		predicted_delta = states[1:] - states[:-1]
		clip = self.cfg.sa_norm_clip
		if clip is None:
			imagined_clip_fraction = torch.zeros((), device=states.device)
			real_clip_fraction = torch.zeros((), device=states.device)
		else:
			imagined_clip_fraction = (states.abs() > clip).float().mean()
			real_clip_fraction = (real_x.abs() > clip).float().mean()
		diagnostics = {
			"state_norm_num_states": self.model.state_norm.num_fit_states.to(
				device=states.device, dtype=states.dtype
			),
			"state_norm_mean_abs": self.model.state_norm.mean.abs().mean(),
			"state_norm_std_mean": self.model.state_norm.std.mean(),
			"state_norm_std_min": self.model.state_norm.std.min(),
			"state_norm_std_max": self.model.state_norm.std.max(),
			"imagined_state_abs_max": raw_states.abs().max(),
			"normalized_imagined_state_abs_max": states.abs().max(),
			"predicted_normalized_delta_abs_mean": predicted_delta.abs().mean(),
			"predicted_normalized_delta_abs_max": predicted_delta.abs().max(),
			"imagined_state_clip_fraction": imagined_clip_fraction,
			"real_state_clip_fraction": real_clip_fraction,
		}
		for step in (1, 3):
			if step < states.shape[0] and step < real_states.shape[0]:
				difference = states[step] - real_x[step]
				diagnostics[f"state_nrmse_{step}"] = (
					difference.square().mean(dim=-1).sqrt().mean()
				)
		return diagnostics

	def _all_replay_states(self, buffer) -> torch.Tensor:
		"""Read every stored replay observation exactly once without sampling."""
		if not hasattr(buffer, "_buffer"):
			raise RuntimeError(
				"State-Anchored normalizer initialization requires a replay wrapper "
				"with a '_buffer' ReplayBuffer field."
			)
		replay = buffer._buffer
		try:
			num_entries = len(replay)
		except TypeError as exc:
			raise RuntimeError(
				"State-Anchored normalizer could not determine replay length."
			) from exc
		if num_entries == 0:
			raise RuntimeError(
				"State-Anchored normalizer cannot initialize from an empty replay."
			)
		try:
			stored = replay[:num_entries]
		except (IndexError, KeyError, TypeError, RuntimeError) as exc:
			raise RuntimeError(
				"State-Anchored normalizer could not index all replay entries "
				"with the installed TorchRL API."
			) from exc
		if not hasattr(stored, "get"):
			raise RuntimeError(
				"State-Anchored replay indexing did not return a TensorDict-like object."
			)
		states = stored.get("obs", None)
		if not isinstance(states, torch.Tensor):
			raise RuntimeError(
				"State-Anchored replay storage does not contain a tensor 'obs' field."
			)
		state_dim = self.cfg.obs_shape["state"][0]
		if states.ndim < 1 or states.shape[-1] != state_dim:
			raise RuntimeError(
				f"Expected replay observations with final dimension {state_dim}, "
				f"got shape {tuple(states.shape)}."
			)
		return states

	def update(self, buffer, log_diagnostics=True):
		"""Initialize/update state statistics, then run one model update."""
		timing = {}
		if not self._norm_initialized:
			_t = perf_counter()
			all_seed_states = self._all_replay_states(buffer)
			self.model.state_norm.fit(all_seed_states)
			self._norm_initialized = True
			timing["state_norm_fit_submit_ms"] = 1e3 * (perf_counter() - _t)

		_t = perf_counter()
		obs, action, reward, terminated, task = buffer.sample()
		timing["replay_sample_submit_ms"] = 1e3 * (perf_counter() - _t)
		if task is not None:
			raise NotImplementedError(
				"State-Anchored TD-MPC2 does not support multitask replay batches."
			)
		if self._sa_update_count_cpu < self.cfg.sa_norm_freeze_updates:
			# Only actual replay observations are allowed to update these statistics.
			_t = perf_counter()
			self.model.state_norm.update(obs)
			timing["state_norm_submit_ms"] = 1e3 * (perf_counter() - _t)
		log_this_step = self.cfg.sa_log_diagnostics and log_diagnostics
		self._sa_update_count.add_(1)
		self._sa_update_count_cpu += 1

		torch.compiler.cudagraph_mark_step_begin()
		_t = perf_counter()
		result = self._update(obs, action, reward, terminated, log_this_step=log_this_step)
		timing["compiled_update_submit_ms"] = 1e3 * (perf_counter() - _t)
		self._last_runtime_timing = timing
		return result

	def _update(self, obs, action, reward, terminated, task=None, log_this_step: bool = False):
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
			if self.cfg.sa_state_loss == "mse":
				step_loss = F.mse_loss(state, target_next_state)
			else:
				step_loss = F.smooth_l1_loss(
					state,
					target_next_state,
					beta=self.cfg.sa_state_loss_beta,
				)
			state_loss = state_loss + step_loss * self.cfg.rho ** t
			states[t + 1] = state

		model_states = states[:-1]
		model_features = self.model.features(model_states)
		qs = self.model.Q(model_states, action, task, return_type="all", _features=model_features)
		reward_preds = self.model.reward(model_states, action, task, _features=model_features)
		if self.cfg.episodic:
			term_features = self.model.features(states[1:])
			termination_pred = self.model.termination(
				states[1:], task, unnormalized=True, _features=term_features
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
			if log_this_step
			else {}
		)
		total_loss.backward()
		if log_this_step:
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
		# Clear any encoder gradients that may have leaked during the actor update.
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

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""Compute TD-target, sharing features between pi and target Q."""
		features = self.model.features(next_z)
		action, _ = self.model.pi(next_z, task, _features=features)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1 - terminated) * self.model.Q(
			next_z, action, task, return_type="min", target=True, _features=features
		)

	@torch.no_grad()
	def _estimate_value(self, state, actions, task, initial_features=None):
		"""Estimate a trajectory using one feature computation per state."""
		self.model._assert_single_task(task)
		value, discount = 0, 1
		termination = torch.zeros(
			self.cfg.num_samples,
			1,
			dtype=torch.float32,
			device=state.device,
		)
		features = (
			initial_features
			if initial_features is not None
			else self.model.features(state)
		)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(
				self.model.reward(state, actions[t], task, _features=features),
				self.cfg,
			)
			state = self.model.next(
				state, actions[t], task, _features=features
			)
			value = value + discount * (1 - termination) * reward
			discount = discount * self.discount
			features = self.model.features(state)
			if self.cfg.episodic:
				termination = torch.clip(
					termination
					+ (
						self.model.termination(
							state, task, _features=features
						)
						> 0.5
					).float(),
					max=1.0,
				)
		action, _ = self.model.pi(state, task, _features=features)
		return value + discount * (1 - termination) * self.model.Q(
			state,
			action,
			task,
			return_type="avg",
			_features=features,
		)

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""Plan with MPPI while sharing features within each imagined state."""
		self.model._assert_single_task(task)
		state = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon,
				self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=self.device,
			)
			pi_state = state.repeat(self.cfg.num_pi_trajs, 1)
			pi_features = self.model.features(pi_state)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t], _ = self.model.pi(
					pi_state, task, _features=pi_features
				)
				pi_state = self.model.next(
					pi_state,
					pi_actions[t],
					task,
					_features=pi_features,
				)
				pi_features = self.model.features(pi_state)
			pi_actions[-1], _ = self.model.pi(
				pi_state, task, _features=pi_features
			)

		state = state.repeat(self.cfg.num_samples, 1)
		initial_features = self.model.features(state)
		mean = torch.zeros(
			self.cfg.horizon, self.cfg.action_dim, device=self.device
		)
		std = torch.full(
			(self.cfg.horizon, self.cfg.action_dim),
			self.cfg.max_std,
			dtype=torch.float,
			device=self.device,
		)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(
			self.cfg.horizon,
			self.cfg.num_samples,
			self.cfg.action_dim,
			device=self.device,
		)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		for _ in range(self.cfg.iterations):
			r = torch.randn(
				self.cfg.horizon,
				self.cfg.num_samples - self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=std.device,
			)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample

			value = self._estimate_value(
				state, actions, task, initial_features=initial_features
			).nan_to_num(0)
			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0
			).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			max_value = elite_value.max(0).values
			score = torch.exp(
				self.cfg.temperature * (elite_value - max_value)
			)
			score = score / score.sum(0)
			mean = (
				(score.unsqueeze(0) * elite_actions).sum(dim=1)
				/ (score.sum(0) + 1e-9)
			)
			std = (
				(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2)
				.sum(dim=1)
				/ (score.sum(0) + 1e-9)
			).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)

		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		action, std = actions[0], std[0]
		if not eval_mode:
			action = action + std * torch.randn(
				self.cfg.action_dim, device=std.device
			)
		self._prev_mean.copy_(mean)
		return action.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""Update policy, computing features once shared between pi and Q."""
		features = self.model.features(zs).detach()
		action, info = self.model.pi(zs, task, _features=features)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True, _features=features)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return {
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		}

	def save(self, fp) -> None:
		"""Save the stabilized carrier version and normalizer freeze counter."""
		torch.save({
			"state_anchored_checkpoint_version": self._CHECKPOINT_VERSION,
			"model": self.model.state_dict(),
			"sa_update_count": self._sa_update_count.detach().cpu(),
		}, fp)

	def load(self, fp) -> None:
		"""Load only checkpoints using the stabilized normalized carrier."""
		if isinstance(fp, dict):
			checkpoint = fp
		else:
			checkpoint = torch.load(
				fp,
				map_location=torch.get_default_device(),
				weights_only=False,
			)
		if not isinstance(checkpoint, dict) or checkpoint.get(
			"state_anchored_checkpoint_version"
		) != self._CHECKPOINT_VERSION:
			raise RuntimeError(
				"Incompatible State-Anchored checkpoint: stabilized normalized-carrier "
				"checkpoint version 2 is required; legacy raw-carrier checkpoints "
				"cannot be loaded."
			)
		model_state = checkpoint.get("model")
		if not isinstance(model_state, dict) or "state_norm.initialized" not in model_state:
			raise RuntimeError(
				"Incompatible State-Anchored checkpoint: normalizer initialization "
				"state is missing."
			)
		if not bool(model_state["state_norm.initialized"].item()):
			raise RuntimeError(
				"Cannot load a State-Anchored checkpoint with an uninitialized "
				"state normalizer."
			)
		if "sa_update_count" not in checkpoint:
			raise RuntimeError(
				"Incompatible State-Anchored checkpoint: normalizer freeze counter "
				"is missing."
			)
		super().load({"model": model_state})
		self._norm_initialized = True
		self._sa_update_count.copy_(torch.as_tensor(
			checkpoint["sa_update_count"],
			device=self._sa_update_count.device,
			dtype=self._sa_update_count.dtype,
		))
		self._sa_update_count_cpu = int(checkpoint["sa_update_count"].item())
