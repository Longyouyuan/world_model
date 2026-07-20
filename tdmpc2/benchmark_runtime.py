"""Controlled runtime benchmark for standard and state-anchored TD-MPC2.

This script intentionally avoids W&B, evaluation, checkpointing, and video
recording. It warms up ``torch.compile`` before measuring steady-state online
training phases and prints both wall-clock and main-thread CPU time.
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import json
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import hydra
import numpy as np
import torch
from tensordict.tensordict import TensorDict

from common.buffer import Buffer
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from state_anchored.agent import StateAnchoredTDMPC2
from state_anchored.config import apply_state_anchored_defaults
from tdmpc2 import TDMPC2


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class PhaseTimer:
	"""Accumulate wall time, current-thread CPU time, and CUDA latency."""

	def __init__(self, name: str):
		self.name = name
		self.wall_seconds: list[float] = []
		self.cpu_seconds: list[float] = []
		self.cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

	@contextmanager
	def measure(self):
		start_event = torch.cuda.Event(enable_timing=True)
		end_event = torch.cuda.Event(enable_timing=True)
		start_event.record()
		wall_start = time.perf_counter()
		cpu_start = time.thread_time()
		with torch.profiler.record_function(f"benchmark/{self.name}"):
			yield
		self.cpu_seconds.append(time.thread_time() - cpu_start)
		self.wall_seconds.append(time.perf_counter() - wall_start)
		end_event.record()
		self.cuda_events.append((start_event, end_event))

	def summary(self) -> dict[str, float | int]:
		cuda_ms = [start.elapsed_time(end) for start, end in self.cuda_events]
		return {
			"calls": len(self.wall_seconds),
			"wall_ms_mean": 1e3 * float(np.mean(self.wall_seconds)),
			"wall_ms_p50": 1e3 * float(np.percentile(self.wall_seconds, 50)),
			"wall_ms_p95": 1e3 * float(np.percentile(self.wall_seconds, 95)),
			"thread_cpu_ms_mean": 1e3 * float(np.mean(self.cpu_seconds)),
			"cuda_interval_ms_mean": float(np.mean(cuda_ms)),
		}

	def clear(self) -> None:
		self.wall_seconds.clear()
		self.cpu_seconds.clear()
		self.cuda_events.clear()


def _to_td(env, obs, action=None, reward=None, terminated=None) -> TensorDict:
	if isinstance(obs, dict):
		obs = TensorDict(obs, batch_size=(), device="cpu")
	else:
		obs = obs.unsqueeze(0).cpu()
	if action is None:
		action = torch.full_like(env.rand_act(), float("nan"))
	if reward is None:
		reward = torch.tensor(float("nan"))
	if terminated is None:
		terminated = torch.tensor(float("nan"))
	return TensorDict(
		obs=obs,
		action=action.unsqueeze(0),
		reward=reward.unsqueeze(0),
		terminated=terminated.unsqueeze(0),
		batch_size=(1,),
	)


def _fill_seed_buffer(cfg, env, buffer: Buffer) -> None:
	"""Collect the same number of random seed steps as the online trainer."""
	collected = 0
	while collected < cfg.seed_steps:
		obs = env.reset()
		tds = [_to_td(env, obs)]
		done = False
		while not done and collected < cfg.seed_steps:
			action = env.rand_act()
			obs, reward, done, info = env.step(action)
			tds.append(_to_td(env, obs, action, reward, info["terminated"]))
			collected += 1
		buffer.add(torch.cat(tds))


def _install_phase_wrappers(agent, buffer: Buffer, phases: dict[str, PhaseTimer]) -> None:
	original_sample = buffer.sample

	def timed_sample():
		with phases["replay_sample"].measure():
			return original_sample()

	buffer.sample = timed_sample
	original_core_update = agent._update

	def timed_core_update(*args, **kwargs):
		with phases["compiled_update"].measure():
			return original_core_update(*args, **kwargs)

	agent._update = timed_core_update
	if isinstance(agent, StateAnchoredTDMPC2):
		original_norm_update = agent.model.state_norm.update

		def timed_norm_update(_self, states):
			with phases["state_norm_update"].measure():
				return original_norm_update(states)

		agent.model.state_norm.update = MethodType(
			timed_norm_update, agent.model.state_norm
		)


def _warm_up(agent, buffer: Buffer, env, obs, warmup_updates: int) -> None:
	"""Compile all branches used by steady-state online training."""
	agent.update(buffer, log_diagnostics=False)
	if isinstance(agent, StateAnchoredTDMPC2):
		agent.update(buffer, log_diagnostics=True)
	for _ in range(warmup_updates):
		agent.update(buffer, log_diagnostics=True)
	for t0 in (True, False):
		action = agent.act(obs, t0=t0, eval_mode=False)
		obs, _, _, _ = env.step(action)
	torch.cuda.synchronize()


def _profiled_steps(agent, buffer, env, obs, steps: int, diagnostics: bool):
	activities = [
		torch.profiler.ProfilerActivity.CPU,
		torch.profiler.ProfilerActivity.CUDA,
	]
	with torch.profiler.profile(
		activities=activities,
		record_shapes=False,
		profile_memory=False,
		with_stack=False,
	) as profile:
		for index in range(steps):
			with torch.profiler.record_function("benchmark/profile_step"):
				action = agent.act(obs, t0=index == 0, eval_mode=False)
				obs, _, done, _ = env.step(action)
				agent.update(buffer, log_diagnostics=diagnostics)
				if done:
					obs = env.reset()
		torch.cuda.synchronize()
	return obs, profile


@hydra.main(config_name="config", config_path=".", version_base=None)
def benchmark(cfg):
	agent_kind = str(cfg.get("benchmark_agent", "standard"))
	if agent_kind not in {"standard", "state_anchored"}:
		raise ValueError("+benchmark_agent must be standard or state_anchored")
	warmup_updates = int(cfg.get("benchmark_warmup_updates", 20))
	baseline_steps = int(cfg.get("benchmark_baseline_steps", 500))
	measure_steps = int(cfg.get("benchmark_measure_steps", 200))
	profile_steps = int(cfg.get("benchmark_profile_steps", 20))
	diagnostics = bool(cfg.get("benchmark_diagnostics", True))
	output = Path(str(cfg.get("benchmark_output", f"benchmark-{agent_kind}.json")))

	cfg = parse_cfg(cfg)
	if agent_kind == "state_anchored":
		cfg = apply_state_anchored_defaults(cfg)
	set_seed(cfg.seed)
	env = make_env(cfg)
	buffer = Buffer(cfg)
	_fill_seed_buffer(cfg, env, buffer)
	agent = StateAnchoredTDMPC2(cfg) if agent_kind == "state_anchored" else TDMPC2(cfg)
	obs = env.reset()

	print(f"Warming {agent_kind} benchmark...")
	_warm_up(agent, buffer, env, obs, warmup_updates)
	obs = env.reset()
	torch.cuda.synchronize()
	baseline_started_unix = time.time()
	baseline_wall_start = time.perf_counter()
	baseline_cpu_start = time.thread_time()
	for index in range(baseline_steps):
		action = agent.act(obs, t0=index == 0, eval_mode=False)
		obs, _, done, _ = env.step(action)
		agent.update(buffer, log_diagnostics=diagnostics)
		if done:
			obs = env.reset()
	torch.cuda.synchronize()
	baseline_ended_unix = time.time()
	baseline_thread_cpu = time.thread_time() - baseline_cpu_start
	baseline_wall = time.perf_counter() - baseline_wall_start

	phases = {
		name: PhaseTimer(name)
		for name in (
			"act",
			"env_step",
			"update_total",
			"replay_sample",
			"state_norm_update",
			"compiled_update",
		)
	}
	_install_phase_wrappers(agent, buffer, phases)
	# Wrapping adds record-function and event boundaries. Warm that exact path,
	# then discard its first-call allocations/captures before phase measurement.
	for index in range(20):
		action = agent.act(obs, t0=False, eval_mode=False)
		obs, _, done, _ = env.step(action)
		agent.update(buffer, log_diagnostics=diagnostics)
		if done:
			obs = env.reset()
	torch.cuda.synchronize()
	for timer in phases.values():
		timer.clear()
	obs = env.reset()
	torch.cuda.synchronize()
	benchmark_wall_start = time.perf_counter()
	benchmark_cpu_start = time.thread_time()
	for index in range(measure_steps):
		with phases["act"].measure():
			action = agent.act(obs, t0=index == 0, eval_mode=False)
		with phases["env_step"].measure():
			obs, _, done, _ = env.step(action)
		with phases["update_total"].measure():
			agent.update(buffer, log_diagnostics=diagnostics)
		if done:
			obs = env.reset()
	torch.cuda.synchronize()
	benchmark_thread_cpu = time.thread_time() - benchmark_cpu_start
	benchmark_wall = time.perf_counter() - benchmark_wall_start
	phase_summaries = {
		name: timer.summary()
		for name, timer in phases.items()
		if timer.wall_seconds
	}

	obs, profile = _profiled_steps(
		agent, buffer, env, obs, profile_steps, diagnostics
	)
	profile_table_cpu = profile.key_averages().table(
		sort_by="self_cpu_time_total", row_limit=30
	)
	profile_table_cuda = profile.key_averages().table(
		sort_by="self_cuda_time_total", row_limit=30
	)
	result = {
		"agent": agent_kind,
		"diagnostics": diagnostics,
		"warmup_updates": warmup_updates,
		"baseline_steps": baseline_steps,
		"baseline_started_unix": baseline_started_unix,
		"baseline_ended_unix": baseline_ended_unix,
		"baseline_wall_seconds": baseline_wall,
		"baseline_thread_cpu_seconds": baseline_thread_cpu,
		"baseline_steps_per_second": baseline_steps / baseline_wall,
		"baseline_thread_cpu_fraction": baseline_thread_cpu / baseline_wall,
		"measure_steps": measure_steps,
		"profile_steps": profile_steps,
		"total_wall_seconds": benchmark_wall,
		"total_thread_cpu_seconds": benchmark_thread_cpu,
		"steps_per_second": measure_steps / benchmark_wall,
		"thread_cpu_fraction": benchmark_thread_cpu / benchmark_wall,
		"phases": phase_summaries,
		"profile_cpu_table": profile_table_cpu,
		"profile_cuda_table": profile_table_cuda,
	}
	output.write_text(json.dumps(result, indent=2))
	print("\n=== Benchmark summary ===")
	print(json.dumps({key: value for key, value in result.items() if not key.startswith("profile_")}, indent=2))
	print("\n=== Top CPU operations ===")
	print(profile_table_cpu)
	print("\n=== Top CUDA operations ===")
	print(profile_table_cuda)
	print(f"\nFull results written to {output.resolve()}")


if __name__ == "__main__":
	benchmark()
