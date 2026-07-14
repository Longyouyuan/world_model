import unittest

import torch

from state_anchored.layers import RunningStateNorm


class RunningStateNormTest(unittest.TestCase):
	def setUp(self):
		self.data = torch.tensor([
			[1.0, 2.0, -1.0],
			[3.0, 4.0, 1.0],
			[5.0, 8.0, 3.0],
			[7.0, 10.0, 5.0],
		], dtype=torch.float32)

	def test_known_mean_and_unbiased_variance(self):
		norm = RunningStateNorm(3)
		norm.update(self.data)
		torch.testing.assert_close(
			norm.mean, self.data.double().mean(0), atol=1e-6, rtol=1e-5
		)
		torch.testing.assert_close(
			norm.var, self.data.double().var(0, unbiased=True),
			atol=1e-6, rtol=1e-5,
		)
		self.assertEqual(norm.count.item(), len(self.data))

	def test_batched_merge_matches_single_update(self):
		one_batch = RunningStateNorm(3)
		two_batches = RunningStateNorm(3)
		one_batch.update(self.data)
		two_batches.update(self.data[:2])
		two_batches.update(self.data[2:])
		torch.testing.assert_close(
			two_batches.mean, one_batch.mean, atol=1e-6, rtol=1e-5
		)
		torch.testing.assert_close(
			two_batches.var, one_batch.var, atol=1e-6, rtol=1e-5
		)
		torch.testing.assert_close(
			two_batches.m2, one_batch.m2, atol=1e-6, rtol=1e-5
		)

	def test_normalize_denormalize_and_dtype(self):
		norm = RunningStateNorm(3, clip=None)
		norm.update(self.data)
		states = self.data.to(torch.float32)
		reconstructed = norm.denormalize(norm.normalize(states))
		self.assertEqual(reconstructed.dtype, states.dtype)
		self.assertEqual(reconstructed.device, states.device)
		torch.testing.assert_close(
			reconstructed, states, atol=1e-6, rtol=1e-5
		)

	def test_min_std_count_fallback_and_delta_is_not_clipped(self):
		norm = RunningStateNorm(2, eps=0.0, min_std=0.25, clip=1.0)
		torch.testing.assert_close(norm.std, torch.ones(2, dtype=torch.float64))
		norm.update(torch.tensor([[2.0, -3.0]]))
		torch.testing.assert_close(norm.std, torch.ones(2, dtype=torch.float64))
		norm.update(torch.tensor([[2.0, -3.0]]))
		self.assertTrue(torch.all(norm.std >= 0.25))
		clipped = norm.normalize(torch.tensor([[100.0, -100.0]]))
		self.assertTrue(torch.all(clipped.abs() <= 1.0))
		scaled = norm.scale_delta(torch.tensor([[20.0, -20.0]]))
		torch.testing.assert_close(scaled.abs(), torch.full_like(scaled, 5.0))

	def test_normalization_does_not_update_statistics(self):
		norm = RunningStateNorm(3)
		norm.update(self.data)
		before = {key: value.clone() for key, value in norm.state_dict().items()}
		predicted_state = torch.randn(4, 3)
		norm.normalize(predicted_state)
		norm.denormalize(predicted_state)
		norm.scale_delta(predicted_state)
		for key, expected in before.items():
			torch.testing.assert_close(norm.state_dict()[key], expected)

	def test_state_dict_roundtrip(self):
		norm = RunningStateNorm(3)
		norm.update(self.data)
		restored = RunningStateNorm(3)
		restored.load_state_dict(norm.state_dict())
		for key, expected in norm.state_dict().items():
			torch.testing.assert_close(restored.state_dict()[key], expected)

	def test_temporal_batch_update(self):
		norm = RunningStateNorm(3)
		states = self.data.reshape(2, 2, 3)
		norm.update(states)
		self.assertEqual(norm.count.item(), 4)
		torch.testing.assert_close(
			norm.mean, self.data.double().mean(0), atol=1e-6, rtol=1e-5
		)


if __name__ == "__main__":
	unittest.main()
