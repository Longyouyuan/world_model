import os
import tempfile
import types
import unittest

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer

from state_anchored.agent import StateAnchoredTDMPC2
from state_anchored.config import apply_state_anchored_defaults


def make_agent_cfg(**overrides):
	values = {
		"obs": "state",
		"multitask": False,
		"obs_shape": {"state": (5,)},
		"action_dim": 2,
		"episode_length": 10,
		"batch_size": 4,
		"horizon": 2,
		"latent_dim": 16,
		"enc_dim": 16,
		"enc_lr_scale": 0.3,
		"num_enc_layers": 2,
		"mlp_dim": 32,
		"simnorm_dim": 8,
		"num_q": 2,
		"num_bins": 11,
		"vmin": -5.0,
		"vmax": 5.0,
		"bin_size": 1.0,
		"dropout": 0.0,
		"episodic": False,
		"log_std_min": -10.0,
		"log_std_max": 2.0,
		"lr": 3e-4,
		"tau": 0.05,
		"grad_clip_norm": 20.0,
		"compile": False,
		"iterations": 1,
		"num_samples": 8,
		"num_elites": 2,
		"num_pi_trajs": 0,
		"min_std": 0.05,
		"max_std": 2.0,
		"temperature": 0.5,
		"mpc": True,
		"entropy_coef": 1e-4,
		"rho": 0.5,
		"reward_coef": 0.1,
		"value_coef": 0.1,
		"termination_coef": 1.0,
		"discount_denom": 5,
		"discount_min": 0.95,
		"discount_max": 0.995,
		"sa_norm_freeze_updates": 2,
		"sa_log_diagnostics": True,
	}
	values.update(overrides)
	return apply_state_anchored_defaults(types.SimpleNamespace(**values))


def make_stored_obs(state_dim=5):
	if state_dim != 5:
		raise ValueError("The deterministic smoke fixture expects state_dim=5.")
	return torch.tensor([
		[-2.0, -1.5, -1.0, -0.5, 0.0],
		[-1.0, -0.5, 0.0, 0.5, 1.0],
		[0.0, 0.5, 1.0, 1.5, 2.0],
		[1.0, 1.5, 2.0, 2.5, 3.0],
		[2.0, 2.5, 3.0, 3.5, 4.0],
		[3.0, 3.5, 4.0, 4.5, 5.0],
		[float("nan"), 20.0, 21.0, 22.0, 23.0],
		[30.0, 31.0, float("inf"), 33.0, 34.0],
	], dtype=torch.float32)


class TorchRLBufferFixture:
	"""Real TorchRL storage plus a deterministic CUDA training minibatch."""

	def __init__(self, cfg, stored_obs=None):
		state_dim = cfg.obs_shape["state"][0]
		if stored_obs is None:
			stored_obs = make_stored_obs(state_dim)
		stored_obs = stored_obs.detach().cpu().clone()
		num_entries = stored_obs.shape[0]
		storage = LazyTensorStorage(max(num_entries + 4, 16), device="cpu")
		self._buffer = ReplayBuffer(storage=storage, batch_size=1)
		self._buffer.extend(TensorDict({
			"obs": stored_obs,
			"action": torch.zeros(num_entries, cfg.action_dim),
			"reward": torch.zeros(num_entries),
			"terminated": torch.zeros(num_entries),
			"episode": torch.zeros(num_entries, dtype=torch.long),
		}, batch_size=[num_entries]))

		device = torch.device("cuda:0")
		obs = torch.linspace(
			-1.25,
			2.25,
			steps=(cfg.horizon + 1) * cfg.batch_size * state_dim,
			device=device,
		).reshape(cfg.horizon + 1, cfg.batch_size, state_dim)
		action = torch.linspace(
			-0.75,
			0.75,
			steps=cfg.horizon * cfg.batch_size * cfg.action_dim,
			device=device,
		).reshape(cfg.horizon, cfg.batch_size, cfg.action_dim).tanh()
		reward = torch.linspace(
			-0.5,
			0.5,
			steps=cfg.horizon * cfg.batch_size,
			device=device,
		).reshape(cfg.horizon, cfg.batch_size, 1)
		terminated = torch.zeros(
			cfg.horizon, cfg.batch_size, 1, device=device
		)
		self.batch = (obs, action, reward, terminated, None)
		self.events = []
		self.sample_calls = 0
		self.before_sample = None

	def sample(self):
		self.events.append("sample")
		self.sample_calls += 1
		if self.before_sample is not None:
			self.before_sample()
		return self.batch


@unittest.skipUnless(torch.cuda.is_available(), "State-Anchored agent requires CUDA")
class StateAnchoredAgentSmokeTest(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(11)
		torch.cuda.manual_seed_all(11)

	def assert_metrics_finite(self, metrics):
		self.assertTrue(metrics)
		for name, value in metrics.items():
			self.assertEqual(value.numel(), 1, name)
			self.assertTrue(torch.isfinite(value).item(), name)

	def test_seed_fit_online_updates_and_freeze(self):
		cfg = make_agent_cfg(sa_norm_freeze_updates=2)
		agent = StateAnchoredTDMPC2(cfg)
		buffer = TorchRLBufferFixture(cfg)
		norm = agent.model.state_norm
		stored_obs = buffer._buffer[:len(buffer._buffer)].get("obs")
		finite_seed = stored_obs[torch.isfinite(stored_obs).all(dim=-1)].double()
		seed_count = finite_seed.shape[0]
		batch_count = (cfg.horizon + 1) * cfg.batch_size

		fit_inputs = []
		fit_snapshot = {}
		original_fit = norm.fit

		def tracked_fit(states):
			buffer.events.append("fit")
			fit_inputs.append(states.detach().cpu().clone())
			return original_fit(states)

		norm.fit = tracked_fit

		def capture_fit_snapshot():
			self.assertTrue(norm.initialized.item())
			if not fit_snapshot:
				fit_snapshot.update({
					"count": norm.count.detach().cpu().clone(),
					"mean": norm.mean.detach().cpu().clone(),
					"m2": norm.m2.detach().cpu().clone(),
					"std": norm.std.detach().cpu().clone(),
					"num_fit_states": norm.num_fit_states.detach().cpu().clone(),
				})

		buffer.before_sample = capture_fit_snapshot

		def record_forward(_module, _inputs):
			self.assertTrue(norm.initialized.item())
			buffer.events.append("forward")

		handle = agent.model._encoder.register_forward_pre_hook(record_forward)
		self.addCleanup(handle.remove)

		model_before = {
			name: parameter.detach().clone()
			for name, parameter in agent.model.named_parameters()
			if parameter.requires_grad
		}
		target_before = agent.model._target_Qs_params["2", "weight"].clone()

		first_metrics = agent.update(buffer)
		self.assert_metrics_finite(first_metrics)
		self.assertIn("state_loss", first_metrics)
		self.assertIn("state_nrmse_1", first_metrics)
		self.assertEqual(buffer.events[0], "fit")
		self.assertLess(buffer.events.index("fit"), buffer.events.index("sample"))
		self.assertLess(buffer.events.index("sample"), buffer.events.index("forward"))
		self.assertEqual(len(fit_inputs), 1)
		torch.testing.assert_close(
			fit_inputs[0], stored_obs.cpu(), equal_nan=True
		)

		expected_mean = finite_seed.mean(dim=0)
		expected_m2 = (finite_seed - expected_mean).square().sum(dim=0)
		expected_std = (
			finite_seed.var(dim=0, unbiased=True) + cfg.sa_norm_eps
		).sqrt().clamp_min(cfg.sa_norm_min_std)
		self.assertEqual(fit_snapshot["count"].item(), seed_count)
		self.assertEqual(fit_snapshot["num_fit_states"].item(), seed_count)
		torch.testing.assert_close(fit_snapshot["mean"], expected_mean)
		torch.testing.assert_close(fit_snapshot["m2"], expected_m2)
		torch.testing.assert_close(fit_snapshot["std"], expected_std)
		self.assertEqual(norm.count.item(), seed_count + batch_count)
		self.assertEqual(norm.num_fit_states.item(), seed_count)
		self.assertEqual(agent._sa_update_count.item(), 1)
		self.assertEqual(
			first_metrics["state_norm_num_states"].item(), seed_count
		)
		self.assertTrue(any(
			not torch.equal(model_before[name], parameter.detach())
			for name, parameter in agent.model.named_parameters()
			if name in model_before
		))
		self.assertFalse(torch.equal(
			target_before,
			agent.model._target_Qs_params["2", "weight"],
		))

		second_metrics = agent.update(buffer)
		self.assert_metrics_finite(second_metrics)
		self.assertEqual(norm.count.item(), seed_count + 2 * batch_count)
		self.assertEqual(agent._sa_update_count.item(), 2)
		self.assertEqual(len(fit_inputs), 1)

		frozen_state = {
			key: value.clone() for key, value in norm.state_dict().items()
		}
		third_metrics = agent.update(buffer)
		self.assert_metrics_finite(third_metrics)
		for key, expected in frozen_state.items():
			torch.testing.assert_close(norm.state_dict()[key], expected)
		self.assertEqual(agent._sa_update_count.item(), 3)
		self.assertEqual(len(fit_inputs), 1)
		self.assertEqual(buffer.sample_calls, 3)

	def test_checkpoint_resume_heads_and_planner_roundtrip(self):
		cfg = make_agent_cfg(sa_norm_freeze_updates=1)
		agent = StateAnchoredTDMPC2(cfg)
		buffer = TorchRLBufferFixture(cfg)
		self.assert_metrics_finite(agent.update(buffer))
		self.assertTrue(agent.model.state_norm.initialized.item())
		self.assertEqual(agent._sa_update_count.item(), 1)

		carrier = agent.model.encode(buffer.batch[0][0])
		state_action = buffer.batch[1][0]
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = os.path.join(directory, "state-anchored.pt")
			agent.save(checkpoint)
			restored = StateAnchoredTDMPC2(cfg)
			restored.load(checkpoint)

			self.assertTrue(restored.model.state_norm.initialized.item())
			self.assertEqual(
				restored._sa_update_count.item(), agent._sa_update_count.item()
			)
			for key, expected in agent.model.state_norm.state_dict().items():
				torch.testing.assert_close(
					restored.model.state_norm.state_dict()[key], expected
				)
			self.assertTrue(torch.equal(
				restored.model._target_Qs_params["2", "weight"],
				agent.model._target_Qs_params["2", "weight"],
			))
			with torch.no_grad():
				torch.testing.assert_close(
					restored.model.next(carrier, state_action),
					agent.model.next(carrier, state_action),
				)
				torch.testing.assert_close(
					restored.model.reward(carrier, state_action),
					agent.model.reward(carrier, state_action),
				)
				torch.testing.assert_close(
					restored.model.Q(carrier, state_action, return_type="all"),
					agent.model.Q(carrier, state_action, return_type="all"),
				)
				torch.testing.assert_close(
					restored.model._pi(restored.model.features(carrier)),
					agent.model._pi(agent.model.features(carrier)),
				)

			planner_obs = torch.linspace(-0.5, 0.5, steps=5)
			torch.manual_seed(123)
			torch.cuda.manual_seed_all(123)
			action = agent.act(planner_obs, t0=True, eval_mode=True)
			torch.manual_seed(123)
			torch.cuda.manual_seed_all(123)
			restored_action = restored.act(planner_obs, t0=True, eval_mode=True)
			self.assertEqual(action.shape, (cfg.action_dim,))
			self.assertTrue(torch.isfinite(action).all())
			torch.testing.assert_close(restored_action, action)

			frozen_state = {
				key: value.clone()
				for key, value in restored.model.state_norm.state_dict().items()
			}
			resume_buffer = TorchRLBufferFixture(
				cfg, stored_obs=torch.zeros(1, cfg.obs_shape["state"][0])
			)

			def fail_if_refit(_states):
				raise AssertionError("restored initialized normalizer was refit")

			restored.model.state_norm.fit = fail_if_refit
			resume_metrics = restored.update(resume_buffer)
			self.assert_metrics_finite(resume_metrics)
			for key, expected in frozen_state.items():
				torch.testing.assert_close(
					restored.model.state_norm.state_dict()[key], expected
				)
			self.assertEqual(restored._sa_update_count.item(), 2)
			self.assertEqual(resume_buffer.sample_calls, 1)

		legacy_agent = StateAnchoredTDMPC2(cfg)
		with self.assertRaisesRegex(RuntimeError, "legacy raw-carrier"):
			legacy_agent.load({"model": agent.model.state_dict()})

		uninitialized_agent = StateAnchoredTDMPC2(cfg)
		uninitialized_checkpoint = {
			"state_anchored_checkpoint_version": agent._CHECKPOINT_VERSION,
			"model": uninitialized_agent.model.state_dict(),
			"sa_update_count": torch.tensor(0),
		}
		with self.assertRaisesRegex(RuntimeError, "uninitialized"):
			uninitialized_agent.load(uninitialized_checkpoint)

	@unittest.skipUnless(
		os.getenv("SA_TEST_COMPILE") == "1",
		"Set SA_TEST_COMPILE=1 to exercise torch.compile",
	)
	def test_compile_true_update(self):
		cfg = make_agent_cfg(compile=True, sa_norm_freeze_updates=1)
		agent = StateAnchoredTDMPC2(cfg)
		buffer = TorchRLBufferFixture(cfg)
		metrics = agent.update(buffer)
		self.assert_metrics_finite(metrics)
		self.assertTrue(agent.model.state_norm.initialized.item())
		self.assertEqual(
			agent.model.state_norm.num_fit_states.item(),
			make_stored_obs().shape[0] - 2,
		)
		self.assertEqual(buffer.sample_calls, 1)


if __name__ == "__main__":
	unittest.main()
