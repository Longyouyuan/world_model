# State-Anchored TD-MPC2

This implementation keeps the official TD-MPC2 planner, distributional reward
and Q heads, Q ensemble and target update, Gaussian policy prior, replay buffer,
trainer, logger, and checkpoint path. Its recursive world-model carrier is the
raw state. At every model step it normalizes that state, recomputes an auxiliary
SimNorm feature, and predicts a normalized state delta.

Current scope is intentionally limited to single-task online RL with
`obs=state`. RGB observations, multitask training, and offline training raise a
clear error.

Run commands from this source directory (`tdmpc2/` inside the repository).

## Training

```bash
python train_state_anchored.py \
  task=walker-walk model_size=5 steps=1000000 \
  exp_name=state-anchored compile=false
```

State-Anchored settings are Hydra additions, for example:

```bash
python train_state_anchored.py \
  task=walker-walk model_size=5 compile=false \
  +sa_state_coef=5.0 +sa_state_loss=smooth_l1
```

## Evaluation

```bash
python evaluate_state_anchored.py \
  task=walker-walk model_size=5 \
  checkpoint=/path/to/state-anchored.pt eval_episodes=10
```

State-Anchored checkpoints are not shape-compatible with official latent-model
checkpoints.

## Tests

```bash
python -m unittest discover -s tests -p 'test_state_anchored_*.py' -v
python -m compileall state_anchored train_state_anchored.py \
  evaluate_state_anchored.py tests
```
