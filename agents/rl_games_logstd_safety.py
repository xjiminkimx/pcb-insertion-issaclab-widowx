"""Clamp rl_games continuous policy log_std before exp() to avoid sigma underflow.

rl_games ModelA2CContinuousLogStd uses sigma = exp(logstd). When the fixed log_std
parameter drifts very negative during optimization, exp() can underflow to 0 in
float32, and torch.distributions.Normal.sample() raises:

    RuntimeError: normal expects all elements of std >= 0.0

This module monkey-patches ModelA2CContinuousLogStd.Network.forward once (idempotent).
"""

from __future__ import annotations

_PATCHED = False


def apply_patch() -> None:
    """Apply the forward patch exactly once."""
    global _PATCHED
    if _PATCHED:
        return
    import torch

    from rl_games.algos_torch.models import ModelA2CContinuousLogStd

    NetworkCls = ModelA2CContinuousLogStd.Network
    if getattr(NetworkCls.forward, "__rlgames_logstd_safe__", False):
        _PATCHED = True
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
    NetworkCls.forward = forward
    _PATCHED = True
