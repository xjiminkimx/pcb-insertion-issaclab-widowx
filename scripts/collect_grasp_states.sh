#!/usr/bin/env bash
# Collect grasp_success terminal states for Slide-phase training.
# Defaults (in collect_grasp_states.py): num_envs=4096, num_states=500, headless.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COLLECT_PY="${SCRIPT_DIR}/collect_grasp_states.py"
CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_grasp/nn"
DEFAULT_CKPT_NAME="widowx_pcb_grasp.pth"

if [[ ! -f "${COLLECT_PY}" ]]; then
  echo "[ERROR] collect_grasp_states.py not found at: ${COLLECT_PY}" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

has_checkpoint=false
for arg in "$@"; do
  if [[ "${arg}" == "--checkpoint" ]]; then
    has_checkpoint=true
    break
  fi
done

cd "${WORKSPACE_DIR}"

if [[ "${has_checkpoint}" == false ]]; then
  if [[ -f "${CKPT_DIR}/${DEFAULT_CKPT_NAME}" ]]; then
    echo "[INFO] Checkpoint: ${CKPT_DIR}/${DEFAULT_CKPT_NAME}"
  else
    echo "[INFO] Checkpoint dir: ${CKPT_DIR}/ (auto-resolve latest last_*.pth in Python)"
  fi
  exec "${PYTHON}" "${COLLECT_PY}" "$@"
else
  echo "[INFO] Checkpoint dir: ${CKPT_DIR}/"
  exec "${PYTHON}" "${COLLECT_PY}" "$@"
fi
