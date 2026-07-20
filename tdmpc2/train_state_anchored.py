import os

os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
# os.environ["TORCH_LOGS"] = "+recompiles"
os.environ.setdefault("TORCH_LOGS", "+recompiles")

import warnings

warnings.filterwarnings("ignore")

import hydra
import torch
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from state_anchored.agent import StateAnchoredTDMPC2
from state_anchored.config import apply_state_anchored_defaults
from trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
	"""Train single-task, online, state-observation State-Anchored TD-MPC2."""
	assert cfg.steps > 0, "Must train for at least 1 step."
	actual_cfg = apply_state_anchored_defaults(parse_cfg(cfg))
	assert torch.cuda.is_available()
	set_seed(actual_cfg.seed)
	print(colored("Work dir:", "yellow", attrs=["bold"]), actual_cfg.work_dir)

	env = make_env(actual_cfg)
	trainer = OnlineTrainer(
		cfg=actual_cfg,
		env=env,
		agent=StateAnchoredTDMPC2(actual_cfg),
		buffer=Buffer(actual_cfg),
		logger=Logger(actual_cfg),
	)
	trainer.train()
	print("\nTraining completed successfully")


if __name__ == "__main__":
	train()
