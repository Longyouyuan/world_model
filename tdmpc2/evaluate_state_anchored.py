import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")

import warnings

warnings.filterwarnings("ignore")

import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from state_anchored.agent import StateAnchoredTDMPC2
from state_anchored.config import apply_state_anchored_defaults

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
	"""Evaluate a single-task State-Anchored TD-MPC2 checkpoint."""
	assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
	actual_cfg = apply_state_anchored_defaults(parse_cfg(cfg))
	assert torch.cuda.is_available()
	set_seed(actual_cfg.seed)
	print(colored(f"Task: {actual_cfg.task}", "blue", attrs=["bold"]))
	print(colored(
		f'Model size: {actual_cfg.get("model_size", "default")}',
		"blue",
		attrs=["bold"],
	))
	print(colored(f"Checkpoint: {actual_cfg.checkpoint}", "blue", attrs=["bold"]))

	env = make_env(actual_cfg)
	agent = StateAnchoredTDMPC2(actual_cfg)
	assert os.path.exists(actual_cfg.checkpoint), (
		f"Checkpoint {actual_cfg.checkpoint} not found! Must be a valid filepath."
	)
	agent.load(actual_cfg.checkpoint)

	print(colored(
		f"Evaluating State-Anchored agent on {actual_cfg.task}:",
		"yellow",
		attrs=["bold"],
	))
	if actual_cfg.save_video:
		video_dir = os.path.join(actual_cfg.work_dir, "videos")
		os.makedirs(video_dir, exist_ok=True)

	episode_rewards, episode_successes = [], []
	for episode in range(actual_cfg.eval_episodes):
		obs, done, episode_reward, step = env.reset(), False, 0, 0
		if actual_cfg.save_video:
			frames = [env.render()]
		while not done:
			action = agent.act(obs, t0=step == 0)
			obs, reward, done, info = env.step(action)
			episode_reward += reward
			step += 1
			if actual_cfg.save_video:
				frames.append(env.render())
		episode_rewards.append(episode_reward)
		episode_successes.append(info["success"])
		if actual_cfg.save_video:
			imageio.mimsave(
				os.path.join(video_dir, f"{actual_cfg.task}-{episode}.mp4"),
				frames,
				fps=15,
			)

	print(colored(
		f"  {actual_cfg.task:<22}"
		f"\tR: {np.mean(episode_rewards):.01f}  "
		f"\tS: {np.mean(episode_successes):.02f}",
		"yellow",
	))


if __name__ == "__main__":
	evaluate()
