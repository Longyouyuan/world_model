"""Configuration defaults and validation for State-Anchored TD-MPC2."""

from __future__ import annotations

from typing import Any


DEFAULTS = {
	# Representation
	"sa_feature_dim": None,
	"sa_feature_hidden_dim": None,
	"sa_feature_lr_scale": None,
	"sa_feature_simnorm": True,
	# State normalization
	"sa_norm_eps": 1e-5,
	"sa_norm_min_std": 1e-3,
	"sa_norm_clip": 10.0,
	"sa_norm_freeze_updates": 100_000,
	# Dynamics
	"sa_predict_delta": True,
	"sa_delta_limit": 5.0,
	"sa_dynamics_hidden_dim": None,
	"sa_dynamics_layers": 2,
	"sa_zero_init_dynamics_output": True,
	# State consistency objective
	"sa_state_loss": "smooth_l1",
	"sa_state_loss_beta": 1.0,
	"sa_state_coef": 5.0,
	# Diagnostics
	"sa_log_diagnostics": True,
}


def _has_field(cfg: Any, name: str) -> bool:
	try:
		return name in cfg
	except TypeError:
		return hasattr(cfg, name)


def _set_field(cfg: Any, name: str, value: Any) -> None:
	try:
		setattr(cfg, name, value)
	except (AttributeError, TypeError):
		cfg[name] = value


def apply_state_anchored_defaults(cfg: Any) -> Any:
	"""Add State-Anchored defaults to a parsed TD-MPC2 config.

	Defaults are only inserted when a field is absent. Fields whose documented
	value is ``None`` are then resolved from the corresponding official model
	configuration.
	"""
	for name, value in DEFAULTS.items():
		if not _has_field(cfg, name):
			_set_field(cfg, name, value)

	if cfg.sa_feature_dim is None:
		cfg.sa_feature_dim = cfg.latent_dim
	if cfg.sa_feature_hidden_dim is None:
		cfg.sa_feature_hidden_dim = cfg.enc_dim
	if cfg.sa_feature_lr_scale is None:
		cfg.sa_feature_lr_scale = cfg.enc_lr_scale
	if cfg.sa_dynamics_hidden_dim is None:
		cfg.sa_dynamics_hidden_dim = cfg.mlp_dim

	if cfg.obs != "state":
		raise NotImplementedError(
			"State-Anchored TD-MPC2 only supports obs=state."
		)
	if cfg.multitask:
		raise NotImplementedError(
			"State-Anchored TD-MPC2 currently supports single-task online RL only."
		)
	if cfg.sa_dynamics_layers < 1:
		raise ValueError("sa_dynamics_layers must be at least 1.")
	if cfg.sa_state_loss not in {"mse", "smooth_l1"}:
		raise ValueError("sa_state_loss must be either 'mse' or 'smooth_l1'.")
	if cfg.sa_norm_min_std <= 0:
		raise ValueError("sa_norm_min_std must be positive.")
	if cfg.sa_norm_eps < 0:
		raise ValueError("sa_norm_eps must be non-negative.")
	if cfg.sa_norm_freeze_updates < 0:
		raise ValueError("sa_norm_freeze_updates must be non-negative.")
	if cfg.sa_feature_dim <= 0 or cfg.sa_feature_hidden_dim <= 0:
		raise ValueError("State feature dimensions must be positive.")
	if cfg.sa_dynamics_hidden_dim <= 0:
		raise ValueError("sa_dynamics_hidden_dim must be positive.")
	if cfg.sa_delta_limit <= 0:
		raise ValueError("sa_delta_limit must be positive.")
	if not cfg.sa_predict_delta:
		raise NotImplementedError(
			"The stabilized State-Anchored model only supports normalized "
			"delta prediction."
		)
	if cfg.sa_state_loss == "smooth_l1" and cfg.sa_state_loss_beta <= 0:
		raise ValueError("sa_state_loss_beta must be positive for smooth_l1.")
	if cfg.sa_feature_simnorm and cfg.sa_feature_dim % cfg.simnorm_dim != 0:
		raise ValueError(
			"sa_feature_dim must be divisible by simnorm_dim when SimNorm is enabled."
		)

	return cfg
