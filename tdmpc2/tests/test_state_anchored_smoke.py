import os
import tempfile
import types
import unittest

import torch

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
		"sa_norm_freeze_updates": 1,
		"sa_log_diagnostics": True,
	}
	values.update(overrides)
	return apply_state_anchored_defaults(types.SimpleNamespace(**values))


class FakeBuffer:
	def __init__(self, cfg):
		device = torch.device("cuda:0")
		self.batch = (
			torch.randn(
				cfg.horizon + 1,
				cfg.batch_size,
				cfg.obs_shape["state"][0],
				device=device,
			),
			torch.randn(
				cfg.horizon,
				cfg.batch_size,
				cfg.action_dim,
				device=device,
			).tanh(),
			torch.randn(
				cfg.horizon, cfg.batch_size, 1, device=device
			),
			torch.zeros(
				cfg.horizon, cfg.batch_size, 1, device=device
			),
			None,
		)

	def sample(self):
		return self.batch


@unittest.skipUnless(torch.cuda.is_available(), "State-Anchored agent requires CUDA")
class StateAnchoredAgentSmokeTest(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(11)
		torch.cuda.manual_seed_all(11)

	def test_update_planner_freeze_and_checkpoint_roundtrip(self):
		cfg = make_agent_cfg()
		agent = StateAnchoredTDMPC2(cfg)
		buffer = FakeBuffer(cfg)
		model_before = {
			name: parameter.detach().clone()
			for name, parameter in agent.model.named_parameters()
			if parameter.requires_grad
		}
		target_before = agent.model._target_Qs_params["2", "weight"].clone()

		metrics = agent.update(buffer)
		self.assertTrue(metrics)
		for name, value in metrics.items():
			self.assertEqual(value.numel(), 1, name)
			self.assertTrue(torch.isfinite(value).item(), name)
		self.assertIn("state_loss", metrics)
		self.assertIn("state_nrmse_1", metrics)
		self.assertTrue(any(
			not torch.equal(model_before[name], parameter.detach())
			for name, parameter in agent.model.named_parameters()
			if name in model_before
		))
		self.assertFalse(torch.equal(
			target_before,
			agent.model._target_Qs_params["2", "weight"],
		))

		expected_count = (cfg.horizon + 1) * cfg.batch_size
		self.assertEqual(agent.model.state_norm.count.item(), expected_count)
		frozen_state = {
			key: value.clone()
			for key, value in agent.model.state_norm.state_dict().items()
		}
		agent.update(buffer)
		for key, expected in frozen_state.items():
			torch.testing.assert_close(
				agent.model.state_norm.state_dict()[key], expected
			)

		action = agent.act(torch.randn(5), t0=True, eval_mode=True)
		self.assertEqual(action.shape, (2,))
		self.assertTrue(torch.isfinite(action).all())

		state = buffer.batch[0][0]
		state_action = buffer.batch[1][0]
		with tempfile.TemporaryDirectory() as directory:
			checkpoint = os.path.join(directory, "state-anchored.pt")
			agent.save(checkpoint)
			restored = StateAnchoredTDMPC2(cfg)
			restored.load(checkpoint)
			for key in ("count", "mean", "m2"):
				self.assertTrue(torch.equal(
					getattr(restored.model.state_norm, key),
					getattr(agent.model.state_norm, key),
				))
			self.assertTrue(torch.equal(
				restored.model._target_Qs_params["2", "weight"],
				agent.model._target_Qs_params["2", "weight"],
			))
			with torch.no_grad():
				torch.testing.assert_close(
					restored.model.next(state, state_action),
					agent.model.next(state, state_action),
				)
				torch.testing.assert_close(
					restored.model.reward(state, state_action),
					agent.model.reward(state, state_action),
				)
				torch.testing.assert_close(
					restored.model.Q(state, state_action, return_type="all"),
					agent.model.Q(state, state_action, return_type="all"),
				)
				torch.testing.assert_close(
					restored.model._pi(restored.model.features(state)),
					agent.model._pi(agent.model.features(state)),
				)

	@unittest.skipUnless(
		os.getenv("SA_TEST_COMPILE") == "1",
		"Set SA_TEST_COMPILE=1 to exercise torch.compile",
	)
	def test_compile_true_update(self):
		cfg = make_agent_cfg(compile=True)
		agent = StateAnchoredTDMPC2(cfg)
		metrics = agent.update(FakeBuffer(cfg))
		self.assertTrue(metrics)
		self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
	unittest.main()
