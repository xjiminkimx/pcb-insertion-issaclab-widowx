"""Safety monkey-patches for rl_games WidowX PCB training.

1. Clamp continuous policy ``log_std`` before ``exp()`` so ``Normal.sample`` never sees
   zero/NaN sigma (underflow of very negative log_std).
2. Sanitize env obs / rewards / episode logs at the ``RlGamesVecEnvWrapper`` boundary.
   ``torch.clamp`` does **not** remove NaN, so a single PhysX blow-up otherwise poisons
   ``RunningMeanStd``, mean episode reward (``rew_nan.pth``), and TensorBoard
   (``x2num.py: NaN or Inf found in input tensor``).
"""

from __future__ import annotations

import math

_PATCHED_POLICY = False
_PATCHED_ENV = False


def apply_patch() -> None:
    """Apply policy + env NaN safety patches (idempotent).

    Env wrapper patch needs ``isaaclab_rl`` (after Kit / pxr). If that import is not ready
    yet, policy patch still applies and env patch is retried on the next ``apply_patch()``
    call (``rl_games_ppo_cfg`` imports this again after AppLauncher).
    """
    _apply_policy_logstd_patch()
    try:
        _apply_env_nan_patch()
    except Exception:  # noqa: BLE001 — Kit may not be up during early package import
        pass


def _apply_policy_logstd_patch() -> None:
    global _PATCHED_POLICY
    if _PATCHED_POLICY:
        return
    import torch

    from rl_games.algos_torch.models import ModelA2CContinuousLogStd

    NetworkCls = ModelA2CContinuousLogStd.Network
    if getattr(NetworkCls.forward, "__rlgames_logstd_safe__", False):
        _PATCHED_POLICY = True
        return

    _orig_forward = NetworkCls.forward

    def forward(self, input_dict):  # noqa: ANN001
        is_train = input_dict.get("is_train", True)
        prev_actions = input_dict.get("prev_actions", None)
        obs = input_dict["obs"]
        if torch.is_tensor(obs):
            obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        input_dict["obs"] = self.norm_obs(obs)
        mu, logstd, value, states = self.a2c_network(input_dict)

        # Replace NaNs from bad physics / obs (avoid poisoning Normal / value loss).
        value = torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)
        logstd = torch.nan_to_num(logstd, nan=0.0, posinf=5.0, neginf=-20.0)

        logstd = torch.clamp(logstd, min=-20.0, max=5.0)
        sigma = torch.exp(logstd)
        sigma = torch.clamp(sigma, min=1e-5, max=100.0)

        distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        if is_train:
            entropy = distr.entropy().sum(dim=-1)
            prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
            return {
                "prev_neglogp": torch.squeeze(prev_neglogp),
                "values": value,
                "entropy": entropy,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }
        selected_action = distr.sample()
        neglogp = self.neglogp(selected_action, mu, sigma, logstd)
        return {
            "neglogpacs": torch.squeeze(neglogp),
            "values": self.denorm_value(value),
            "actions": selected_action,
            "rnn_states": states,
            "mus": mu,
            "sigmas": sigma,
        }

    forward.__rlgames_logstd_safe__ = True  # type: ignore[attr-defined]
    # Keep a reference so linters/tools see the original is intentionally replaced.
    del _orig_forward
    NetworkCls.forward = forward
    _PATCHED_POLICY = True


def _sanitize_extras(extras: dict) -> dict:
    """Replace non-finite scalars in episode / direct log dicts."""
    for key in ("episode", "log"):
        payload = extras.get(key)
        if not isinstance(payload, dict):
            continue
        cleaned = {}
        for k, v in payload.items():
            cleaned[k] = _sanitize_log_value(v)
        extras[key] = cleaned
    return extras


def _sanitize_log_value(v):
    import torch

    if torch.is_tensor(v):
        if v.numel() == 0:
            return v
        if v.ndim == 0 or v.numel() == 1:
            x = float(v.detach().float().reshape(-1)[0].item())
            return 0.0 if not math.isfinite(x) else x
        return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if isinstance(v, (float, int)):
        x = float(v)
        return 0.0 if not math.isfinite(x) else x
    return v


def _apply_env_nan_patch() -> None:
    """nan_to_num obs/rewards before rl_games RunningMeanStd / score meters see them."""
    global _PATCHED_ENV
    if _PATCHED_ENV:
        return
    import torch

    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    if getattr(RlGamesVecEnvWrapper, "__widowx_nan_safe__", False):
        _PATCHED_ENV = True
        return

    _orig_process_obs = RlGamesVecEnvWrapper._process_obs
    _orig_step = RlGamesVecEnvWrapper.step

    def _process_obs(self, obs_dict):  # noqa: ANN001
        # Clamp alone preserves NaN; sanitize first so RunningMeanStd stays finite.
        clip = float(self._clip_obs)
        for key, obs in list(obs_dict.items()):
            if torch.is_tensor(obs):
                obs_dict[key] = torch.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        return _orig_process_obs(self, obs_dict)

    def step(self, actions):  # noqa: ANN001
        obs_and_states, rew, dones, extras = _orig_step(self, actions)
        if torch.is_tensor(rew):
            # Zero non-finite rewards so mean episode score cannot become NaN.
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(extras, dict):
            extras = _sanitize_extras(extras)
        # Obs path already sanitized inside patched ``_process_obs`` (called by ``_orig_step``).
        if torch.is_tensor(obs_and_states):
            clip = float(self._clip_obs)
            obs_and_states = torch.nan_to_num(
                obs_and_states, nan=0.0, posinf=clip, neginf=-clip
            )
        elif isinstance(obs_and_states, dict):
            clip = float(self._clip_obs)

            def _clean(o):
                if torch.is_tensor(o):
                    return torch.nan_to_num(o, nan=0.0, posinf=clip, neginf=-clip)
                if isinstance(o, dict):
                    return {k: _clean(v) for k, v in o.items()}
                return o

            obs_and_states = _clean(obs_and_states)
        return obs_and_states, rew, dones, extras

    RlGamesVecEnvWrapper._process_obs = _process_obs
    RlGamesVecEnvWrapper.step = step
    RlGamesVecEnvWrapper.__widowx_nan_safe__ = True  # type: ignore[attr-defined]
    _PATCHED_ENV = True
