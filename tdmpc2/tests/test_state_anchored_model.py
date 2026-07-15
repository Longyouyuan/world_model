import types
import unittest

import torch
import torch.nn.functional as F

from state_anchored.config import apply_state_anchored_defaults
from state_anchored.world_model import StateAnchoredWorldModel


def make_cfg(**overrides):
	values = {
		"obs": "state",
		"multitask": False,
		"obs_shape": {"state": (5,)},
		"action_dim": 2,
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
		"tau": 0.01,
	}
	values.update(overrides)
	return apply_state_anchored_defaults(types.SimpleNamespace(**values))


class StateAnchoredWorldModelTest(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(7)
		self.cfg = make_cfg()
		self.model = StateAnchoredWorldModel(self.cfg)
		self.state = torch.randn(4, 5)
		self.action = torch.randn(4, 2).tanh()

	def assert_finite(self, *tensors):
		for tensor in tensors:
			self.assertTrue(torch.isfinite(tensor).all().item())

	def test_encode_is_unclipped_and_decode_roundtrips(self):
		self.model.state_norm.update(torch.tensor([
			[-2.0, -4.0, -6.0, -8.0, -10.0],
			[0.0, 0.0, 0.0, 0.0, 0.0],
			[2.0, 4.0, 6.0, 8.0, 10.0],
		]))
		target_carrier = torch.tensor([
			[20.0, -20.0, 3.0, -4.0, 0.5],
			[-15.0, 12.0, -2.0, 1.0, -0.5],
		])
		raw_state = self.model.decode_state(target_carrier)
		encoded = self.model.encode(raw_state)

		torch.testing.assert_close(encoded, target_carrier, atol=2e-5, rtol=1e-6)
		torch.testing.assert_close(
			self.model.decode_state(encoded), raw_state, atol=2e-5, rtol=1e-6
		)
		self.assertGreater(encoded.abs().max().item(), self.cfg.sa_norm_clip)

	def test_features_and_head_shapes_use_normalized_carrier(self):
		carrier = self.model.encode(self.state)
		features = self.model.features(carrier)
		next_state = self.model.next(carrier, self.action)
		reward = self.model.reward(carrier, self.action)
		policy_action, policy_info = self.model.pi(carrier)
		q_all = self.model.Q(carrier, self.action, return_type="all")

		self.assertEqual(features.shape, (4, 21))
		self.assertEqual(next_state.shape, self.state.shape)
		self.assertEqual(reward.shape, (4, 11))
		self.assertEqual(policy_action.shape, (4, 2))
		self.assertEqual(policy_info["mean"].shape, (4, 2))
		self.assertEqual(policy_info["entropy"].shape, (4, 1))
		self.assertEqual(q_all.shape, (2, 4, 11))
		self.assertEqual(
			self.model._Qs.params["2", "weight"].shape,
			(2, 11, 32),
		)
		self.assertFalse(self.model._target_Qs.training)
		self.model.train()
		self.assertFalse(self.model._target_Qs.training)
		self.model.to("cpu")
		self.assertFalse(self.model._target_Qs.training)

		episodic_model = StateAnchoredWorldModel(make_cfg(episodic=True))
		episodic_carrier = episodic_model.encode(self.state)
		termination = episodic_model.termination(episodic_carrier)
		termination_logits = episodic_model.termination(
			episodic_carrier, unnormalized=True
		)
		self.assertEqual(termination.shape, (4, 1))
		self.assertEqual(termination_logits.shape, (4, 1))
		self.assertTrue(torch.all((termination >= 0) & (termination <= 1)).item())
		self.assert_finite(
			features, next_state, reward, policy_action, q_all,
			policy_info["log_std"], policy_info["scaled_entropy"], termination,
			termination_logits,
		)

	def test_zero_initialized_dynamics_is_identity(self):
		carrier = torch.randn(4, 5) * 25
		predicted = self.model.next(carrier, self.action)
		torch.testing.assert_close(predicted, carrier, atol=1e-7, rtol=0)
		self.assertGreater(carrier.abs().max().item(), self.cfg.sa_norm_clip)
		self.assertEqual(self.model.state_norm.count.item(), 0)

	def test_bounded_delta_uses_limit_scaled_tanh(self):
		limit = self.cfg.sa_delta_limit
		raw_delta = torch.tensor([1000.0, -1000.0, limit, -limit, 0.0])
		with torch.no_grad():
			self.model._dynamics.output_layer.weight.zero_()
			self.model._dynamics.output_layer.bias.copy_(raw_delta)

		carrier = torch.randn(4, 5)
		predicted = self.model.next(carrier, self.action)
		delta = predicted - carrier
		expected = limit * torch.tanh(raw_delta / limit)
		torch.testing.assert_close(
			delta, expected.expand_as(delta), atol=1e-6, rtol=1e-6
		)
		self.assertLessEqual(delta.abs().max().item(), limit + 1e-6)

	def test_encoder_input_is_clipped_but_carrier_is_not(self):
		carrier = torch.tensor([
			[20.0, -30.0, 9.0, -9.0, 0.0],
			[-11.0, 12.0, 1.0, -1.0, 3.0],
		])
		carrier_before = carrier.clone()
		seen_inputs = []
		handle = self.model._encoder.register_forward_pre_hook(
			lambda _module, inputs: seen_inputs.append(inputs[0].detach().clone())
		)
		try:
			predicted = self.model.next(carrier, self.action[:2])
		finally:
			handle.remove()

		expected_input = carrier.clamp(-self.cfg.sa_norm_clip, self.cfg.sa_norm_clip)
		self.assertEqual(len(seen_inputs), 1)
		torch.testing.assert_close(seen_inputs[0], expected_input)
		torch.testing.assert_close(predicted, carrier)
		torch.testing.assert_close(carrier, carrier_before)
		self.assertGreater(carrier.abs().max().item(), self.cfg.sa_norm_clip)

	def test_temporal_rollout_is_finite_and_recomputes_features(self):
		model = StateAnchoredWorldModel(make_cfg(
			sa_zero_init_dynamics_output=False
		))
		actions = torch.randn(6, 4, 2).tanh()
		state = model.encode(torch.randn(4, 5))
		states = [state]
		calls = []
		handle = model._encoder.register_forward_hook(
			lambda _module, _inputs, _output: calls.append(1)
		)
		try:
			for action in actions.unbind(0):
				state = model.next(state, action)
				states.append(state)
		finally:
			handle.remove()
		states = torch.stack(states)
		self.assertEqual(len(calls), len(actions))
		self.assertLessEqual(
			(states[1:] - states[:-1]).abs().max().item(),
			model.cfg.sa_delta_limit + 1e-6,
		)

		model_states = states[:-1]
		features = model.features(model_states)
		reward = model.reward(model_states, actions)
		q_all = model.Q(model_states, actions, return_type="all")
		self.assertEqual(features.shape, (6, 4, 21))
		self.assertEqual(reward.shape, (6, 4, 11))
		self.assertEqual(q_all.shape, (2, 6, 4, 11))
		self.assert_finite(states, features, reward, q_all)

	def test_unclipped_state_loss_has_gradient_beyond_clip(self):
		carrier = torch.full((4, 5), 20.0, requires_grad=True)
		predicted = self.model.next(carrier, self.action)
		loss = F.mse_loss(predicted, torch.zeros_like(predicted))
		torch.testing.assert_close(loss, torch.tensor(400.0))
		loss.backward()
		torch.testing.assert_close(carrier.grad, torch.full_like(carrier, 2.0))
		self.assertTrue(
			(self.model._dynamics.output_layer.bias.grad.abs() > 0).all().item()
		)

	def test_state_loss_gradients_reach_feature_encoder_and_dynamics(self):
		model = StateAnchoredWorldModel(make_cfg(
			sa_feature_simnorm=False,
			sa_zero_init_dynamics_output=False,
		))
		carrier = model.encode(torch.randn(4, 5))
		action = torch.randn(4, 2).tanh()
		predicted = model.next(carrier, action)
		target = predicted.detach() + 1.0
		loss = F.mse_loss(predicted, target)
		loss.backward()
		encoder_grads = [
			parameter.grad for parameter in model._encoder.parameters()
			if parameter.grad is not None
		]
		dynamics_grads = [
			parameter.grad for parameter in model._dynamics.parameters()
			if parameter.grad is not None
		]
		self.assertTrue(encoder_grads)
		self.assertTrue(dynamics_grads)
		self.assertTrue(any(gradient.abs().sum() > 0 for gradient in encoder_grads))
		self.assertTrue(any(gradient.abs().sum() > 0 for gradient in dynamics_grads))

	def test_configuration_rejects_unsupported_modes(self):
		with self.assertRaisesRegex(NotImplementedError, "obs=state"):
			make_cfg(obs="rgb")
		with self.assertRaisesRegex(NotImplementedError, "single-task"):
			make_cfg(multitask=True)
		with self.assertRaisesRegex(NotImplementedError, "normalized delta"):
			make_cfg(sa_predict_delta=False)
		for limit in (0.0, -1.0):
			with self.subTest(sa_delta_limit=limit):
				with self.assertRaisesRegex(ValueError, "sa_delta_limit"):
					make_cfg(sa_delta_limit=limit)


if __name__ == "__main__":
	unittest.main()
