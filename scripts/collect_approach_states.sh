#!/usr/bin/env bash
# Collect approach-phase (straddle) terminal states for Insert training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

DEFAULT_CKPT="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_approach/nn/widowx_pcb_approach.pth"
LEGACY_CKPT="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_push/nn/widowx_pcb_push.pth"
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -f "${DEFAULT_CKPT}" ]]; then
    CHECKPOINT="${DEFAULT_CKPT}"
  else
    CHECKPOINT="${LEGACY_CKPT}"
  fi
fi

cd "${WORKSPACE_DIR}"
exec "${PYTHON}" "${SCRIPT_DIR}/collect_approach_states.py" --checkpoint "${CHECKPOINT}" "$@"
