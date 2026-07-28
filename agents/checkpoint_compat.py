"""Checkpoint compatibility helpers for WidowX PCB approach policy play.

Legacy approach checkpoints were trained with 43-dim policy observations.  The
``straddle_closedness`` term (1 dim) was inserted after ``trailing_edge_error``
(at flat index 27), giving 44 dims today.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch

# Policy observation flat layout (Approach, Isaac-WidowX-PCB-Approach-v0).
APPROACH_POLICY_OBS_DIM_CURRENT = 44
APPROACH_POLICY_OBS_DIM_BEFORE_STRADDLE_CLOSEDNESS = 43
# Sum of obs term dims before ``straddle_closedness`` in ObservationsCfg.policy.
APPROACH_OBS_INSERT_STRADDLE_CLOSEDNESS_IDX = 27

_RMS_MEAN_KEY = "running_mean_std.running_mean"
_RMS_VAR_KEY = "running_mean_std.running_var"
_ACTOR_IN_KEY = "a2c_network.actor_mlp.0.weight"

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".checkpoint_cache")


def checkpoint_policy_obs_dim(checkpoint_path: str) -> int | None:
    """Return policy input dim stored in a rl-games checkpoint, or ``None``."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ckpt.get("model")
    if not isinstance(model, dict) or _RMS_MEAN_KEY not in model:
        return None
    return int(model[_RMS_MEAN_KEY].shape[0])


def _insert_obs_dim_1d(
    tensor: torch.Tensor,
    insert_at: int,
    *,
    fill: float,
) -> torch.Tensor:
    """Insert one element into a 1-D running-stat vector."""
    left = tensor[:insert_at]
    mid = tensor.new_tensor([fill])
    right = tensor[insert_at:]
    return torch.cat([left, mid, right], dim=0)


def _insert_obs_dim_actor_in(
    weight: torch.Tensor,
    insert_at: int,
) -> torch.Tensor:
    """Insert a zero input column into the first actor MLP layer."""
    zero_col = torch.zeros(weight.shape[0], 1, dtype=weight.dtype, device=weight.device)
    return torch.cat([weight[:, :insert_at], zero_col, weight[:, insert_at:]], dim=1)


def expand_approach_checkpoint_obs_dim(
    checkpoint: dict[str, Any],
    *,
    target_dim: int = APPROACH_POLICY_OBS_DIM_CURRENT,
    insert_at: int = APPROACH_OBS_INSERT_STRADDLE_CLOSEDNESS_IDX,
    new_obs_mean: float = 0.0,
    new_obs_var: float = 1.0,
) -> dict[str, Any]:
    """Return a copy of ``checkpoint`` with policy obs expanded by one dim at ``insert_at``."""
    model = checkpoint["model"]
    src_dim = int(model[_RMS_MEAN_KEY].shape[0])
    if src_dim == target_dim:
        return checkpoint
    if src_dim + 1 != target_dim:
        raise ValueError(
            f"Unsupported obs expansion {src_dim} -> {target_dim} "
            f"(only +1 at index {insert_at} is implemented)."
        )

    out = dict(checkpoint)
    out_model = dict(model)
    out_model[_RMS_MEAN_KEY] = _insert_obs_dim_1d(
        model[_RMS_MEAN_KEY], insert_at, fill=new_obs_mean
    )
    out_model[_RMS_VAR_KEY] = _insert_obs_dim_1d(
        model[_RMS_VAR_KEY], insert_at, fill=new_obs_var
    )
    out_model[_ACTOR_IN_KEY] = _insert_obs_dim_actor_in(model[_ACTOR_IN_KEY], insert_at)
    out["model"] = out_model
    return out


def _cache_path(source_path: str, target_dim: int, insert_at: int) -> str:
    st = os.stat(source_path)
    digest = hashlib.sha256(
        f"{os.path.abspath(source_path)}:{st.st_mtime_ns}:{st.st_size}:{target_dim}:{insert_at}".encode()
    ).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(_CACHE_DIR, f"{base}_obs{target_dim}_{digest}.pth")


def ensure_approach_checkpoint_compatible(
    checkpoint_path: str,
    *,
    target_dim: int = APPROACH_POLICY_OBS_DIM_CURRENT,
    insert_at: int = APPROACH_OBS_INSERT_STRADDLE_CLOSEDNESS_IDX,
) -> str:
    """Return ``checkpoint_path`` or a cached patched copy loadable by the current env."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    src_dim = checkpoint_policy_obs_dim(checkpoint_path)
    if src_dim is None:
        return checkpoint_path
    if src_dim == target_dim:
        return checkpoint_path
    if src_dim != APPROACH_POLICY_OBS_DIM_BEFORE_STRADDLE_CLOSEDNESS:
        raise ValueError(
            f"Checkpoint obs dim {src_dim} is not compatible with current approach env ({target_dim}). "
            "Only 43 -> 44 (straddle_closedness insert) is supported."
        )

    os.makedirs(_CACHE_DIR, exist_ok=True)
    cached = _cache_path(checkpoint_path, target_dim, insert_at)
    if os.path.isfile(cached) and os.path.getmtime(cached) >= os.path.getmtime(checkpoint_path):
        return cached

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    patched = expand_approach_checkpoint_obs_dim(
        ckpt,
        target_dim=target_dim,
        insert_at=insert_at,
    )
    torch.save(patched, cached)
    return cached


# Deprecated aliases (pre-approach rename).
PUSH_POLICY_OBS_DIM_CURRENT = APPROACH_POLICY_OBS_DIM_CURRENT
PUSH_POLICY_OBS_DIM_BEFORE_STRADDLE_CLOSEDNESS = APPROACH_POLICY_OBS_DIM_BEFORE_STRADDLE_CLOSEDNESS
PUSH_OBS_INSERT_STRADDLE_CLOSEDNESS_IDX = APPROACH_OBS_INSERT_STRADDLE_CLOSEDNESS_IDX
expand_push_checkpoint_obs_dim = expand_approach_checkpoint_obs_dim
ensure_push_checkpoint_compatible = ensure_approach_checkpoint_compatible
