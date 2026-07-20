from time import perf_counter, time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		runtime_times = {}
		cuda_event_pairs = {"act_cuda_interval_ms": [], "update_cuda_interval_ms": []}
		cuda_sample_freq = max(1, int(self.cfg.get("runtime_cuda_sample_freq", 10)))

		def record_timing(key, elapsed_seconds):
			runtime_times.setdefault(key, []).append(1e3 * elapsed_seconds)

		def flush_runtime_metrics(metrics):
			# Resolve sampled CUDA events once per episode instead of synchronizing
			# the training stream on every step.
			if any(cuda_event_pairs.values()):
				torch.cuda.synchronize()
				for key, pairs in cuda_event_pairs.items():
					runtime_times.setdefault(key, []).extend(
						start.elapsed_time(end) for start, end in pairs
					)
					pairs.clear()

			metric_keys = {
				"act_ms", "env_ms", "upd_ms", "act_wall_ms", "env_wall_ms",
				"to_td_wall_ms", "update_submit_wall_ms", "online_step_wall_ms",
				"act_cuda_interval_ms", "update_cuda_interval_ms",
				"replay_sample_submit_ms", "state_norm_fit_submit_ms",
				"state_norm_submit_ms", "compiled_update_submit_ms",
				"buffer_add_wall_ms", "logger_wall_ms", "env_reset_wall_ms",
				"reset_to_td_wall_ms",
			}
			for key in metric_keys:
				metrics.pop(key, None)
			for key, values in runtime_times.items():
				if values:
					metrics[key] = float(np.mean(values))
				values.clear()
			# Preserve the short names already used by the console logger.
			if "act_wall_ms" in metrics:
				metrics["act_ms"] = metrics["act_wall_ms"]
			if "env_wall_ms" in metrics:
				metrics["env_ms"] = metrics["env_wall_ms"]
			if "update_submit_wall_ms" in metrics:
				metrics["upd_ms"] = metrics["update_submit_wall_ms"]

		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					_t = perf_counter()
					eval_metrics = self.eval()
					eval_metrics["eval_wall_ms"] = 1e3 * (perf_counter() - _t)
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					flush_runtime_metrics(train_metrics)
					train_metrics.update(self.common_metrics())
					_t = perf_counter()
					self.logger.log(train_metrics, 'train')
					record_timing("logger_wall_ms", perf_counter() - _t)
					_t = perf_counter()
					self._ep_idx = self.buffer.add(torch.cat(self._tds))
					record_timing("buffer_add_wall_ms", perf_counter() - _t)

				_t = perf_counter()
				obs = self.env.reset()
				record_timing("env_reset_wall_ms", perf_counter() - _t)
				_t = perf_counter()
				self._tds = [self.to_td(obs)]
				record_timing("reset_to_td_wall_ms", perf_counter() - _t)

			# Collect experience
			_step_t = perf_counter()
			if self._step > self.cfg.seed_steps:
				sample_cuda = self._step % cuda_sample_freq == 0
				if sample_cuda:
					act_start, act_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
					act_start.record()
				_t = perf_counter()
				action = self.agent.act(obs, t0=len(self._tds)==1)
				record_timing("act_wall_ms", perf_counter() - _t)
				if sample_cuda:
					act_end.record()
					cuda_event_pairs["act_cuda_interval_ms"].append((act_start, act_end))
			else:
				action = self.env.rand_act()
			_t = perf_counter()
			obs, reward, done, info = self.env.step(action)
			record_timing("env_wall_ms", perf_counter() - _t)
			_t = perf_counter()
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))
			record_timing("to_td_wall_ms", perf_counter() - _t)

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for update_idx in range(num_updates):
					sample_cuda = num_updates == 1 and self._step % cuda_sample_freq == 0
					if sample_cuda:
						update_start, update_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
						update_start.record()
					_t = perf_counter()
					_train_metrics = self.agent.update(
						self.buffer,
						log_diagnostics=(update_idx == num_updates - 1),
					)
					update_elapsed = perf_counter() - _t
					if num_updates == 1:
						record_timing("update_submit_wall_ms", update_elapsed)
					if sample_cuda:
						update_end.record()
						cuda_event_pairs["update_cuda_interval_ms"].append((update_start, update_end))
					if num_updates == 1:
						for key, value in getattr(self.agent, "_last_runtime_timing", {}).items():
							runtime_times.setdefault(key, []).append(value)
				train_metrics.update(_train_metrics)

			if self._step > self.cfg.seed_steps:
				record_timing("online_step_wall_ms", perf_counter() - _step_t)
			self._step += 1

		self.logger.finish(self.agent)
