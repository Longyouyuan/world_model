import unittest

import torch
import torch.nn.functional as F

from state_anchored.layers import RunningStateNorm


class RunningStateNormTest(unittest.TestCase):
	def setUp(self):
		self.data = torch.tensor([
			[1.0, 2.0, -1.0],
			[3.0, 4.0, 1.0],
			[5.0, 8.0, 3.0],
			[7.0, 10.0, 5.0],
		], dtype=torch.float32)

	def assert_exact_statistics(self, norm, states):
		expected = states.double().reshape(-1, states.shape[-1])
		self.assertEqual(norm.count.item(), expected.shape[0])
		torch.testing.assert_close(
			norm.mean, expected.mean(0), atol=1e-12, rtol=1e-12
		)
		torch.testing.assert_close(
			norm.var, expected.var(0, unbiased=True), atol=1e-12, rtol=1e-12
		)

	def test_fit_matches_exact_mean_and_unbiased_variance(self):
		norm = RunningStateNorm(3)
		self.assertFalse(norm.initialized.item())

		norm.fit(self.data)

		self.assertTrue(norm.initialized.item())
		self.assert_exact_statistics(norm, self.data)

	def test_fit_resets_existing_statistics(self):
		norm = RunningStateNorm(3)
		norm.fit(self.data)
		norm.update(torch.full((7, 3), 1000.0))
		self.assertEqual(norm.count.item(), len(self.data) + 7)

		replacement = torch.tensor([
			[-4.0, 3.0, 2.0],
			[0.0, 5.0, 6.0],
			[4.0, 7.0, 10.0],
		], dtype=torch.float32)
		norm.fit(replacement)

		self.assertTrue(norm.initialized.item())
		self.assert_exact_statistics(norm, replacement)
		expected_mean = replacement.double().mean(0)
		expected_m2 = (
			replacement.double() - expected_mean
		).square().sum(0)
		torch.testing.assert_close(norm.m2, expected_m2, atol=1e-12, rtol=1e-12)

	def test_fit_filters_nonfinite_rows(self):
		states = torch.tensor([
			[1.0, 2.0, 3.0],
			[float("nan"), 4.0, 5.0],
			[6.0, float("inf"), 7.0],
			[9.0, 10.0, 11.0],
			[12.0, 13.0, float("-inf")],
		], dtype=torch.float32)
		finite_states = states[[0, 3]]
		norm = RunningStateNorm(3)

		norm.fit(states)

		self.assertTrue(norm.initialized.item())
		self.assert_exact_statistics(norm, finite_states)
		self.assertTrue(torch.isfinite(norm.mean).all().item())
		self.assertTrue(torch.isfinite(norm.m2).all().item())

	def test_fit_requires_two_finite_states(self):
		norm = RunningStateNorm(3)
		states = torch.tensor([
			[1.0, 2.0, 3.0],
			[float("nan"), 4.0, 5.0],
		])
		with self.assertRaises(ValueError):
			norm.fit(states)
		self.assertFalse(norm.initialized.item())

	def test_unclipped_and_network_input_normalization_are_separate(self):
		norm = RunningStateNorm(3, clip=10.0)
		norm.fit(self.data)
		expected_unclipped = torch.tensor([
			[20.0, -15.0, 0.5],
		], dtype=torch.float32)
		mean = norm.mean.to(dtype=expected_unclipped.dtype)
		std = norm.std.to(dtype=expected_unclipped.dtype)
		raw_state = mean + std * expected_unclipped

		unclipped = norm.normalize_unclipped(raw_state)
		for_input = norm.normalize_for_input(raw_state)
		directly_clipped = norm.clip_normalized(unclipped)

		self.assertEqual(unclipped.dtype, raw_state.dtype)
		self.assertEqual(unclipped.device, raw_state.device)
		torch.testing.assert_close(
			unclipped, expected_unclipped, atol=1e-5, rtol=1e-5
		)
		torch.testing.assert_close(
			for_input,
			expected_unclipped.clamp(-10.0, 10.0),
			atol=1e-6,
			rtol=0,
		)
		torch.testing.assert_close(for_input, directly_clipped)
		torch.testing.assert_close(
			norm.denormalize(unclipped), raw_state, atol=1e-6, rtol=1e-5
		)

	def test_clip_none_preserves_normalized_input(self):
		norm = RunningStateNorm(3, clip=None)
		value = torch.tensor([[20.0, -30.0, 40.0]])
		self.assertIs(norm.clip_normalized(value), value)

	def test_unclipped_loss_beyond_clip_has_nonzero_gradient(self):
		norm = RunningStateNorm(1, eps=0.0, clip=10.0)
		norm.fit(torch.tensor([[-1.0], [1.0]]))
		predicted_carrier = torch.tensor([[20.0]], requires_grad=True)
		predicted_raw_state = norm.denormalize(predicted_carrier)
		predicted_x = norm.normalize_unclipped(predicted_raw_state)
		target_x = torch.zeros_like(predicted_x)

		loss = F.mse_loss(predicted_x, target_x)
		loss.backward()

		torch.testing.assert_close(loss.detach(), torch.tensor(400.0))
		torch.testing.assert_close(
			predicted_carrier.grad, torch.tensor([[40.0]])
		)
		self.assertGreater(
			loss.item(),
			F.mse_loss(predicted_x.detach().clamp(-10.0, 10.0), target_x).item(),
		)

	def test_normalization_does_not_update_statistics(self):
		norm = RunningStateNorm(3)
		norm.fit(self.data)
		before = {key: value.clone() for key, value in norm.state_dict().items()}
		predicted_state = torch.randn(4, 3)
		normalized = norm.normalize_unclipped(predicted_state)
		norm.normalize_for_input(predicted_state)
		norm.clip_normalized(normalized)
		norm.denormalize(normalized)
		for key, expected in before.items():
			torch.testing.assert_close(norm.state_dict()[key], expected)

	def test_state_dict_roundtrip_restores_all_buffers(self):
		norm = RunningStateNorm(3)
		norm.fit(self.data)
		norm.update(torch.randn(2, 3))
		state_dict = norm.state_dict()
		self.assertTrue({"initialized", "count", "mean", "m2"}.issubset(state_dict))

		restored = RunningStateNorm(3)
		restored.load_state_dict(state_dict)

		self.assertEqual(set(restored.state_dict()), set(state_dict))
		for key, expected in state_dict.items():
			torch.testing.assert_close(restored.state_dict()[key], expected)
		self.assertTrue(restored.initialized.item())

	def test_temporal_minibatch_update_continues_fit_statistics(self):
		norm = RunningStateNorm(3)
		norm.fit(self.data[:2])
		norm.update(self.data[2:].reshape(1, 2, 3))

		self.assert_exact_statistics(norm, self.data)
		self.assertTrue(norm.initialized.item())


if __name__ == "__main__":
	unittest.main()
