#!/usr/bin/env bash
# Train Approach phase; TensorBoard + checkpoints in logs/rl_games/widowx_pcb_approach/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
TRAIN_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/train.py"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] Isaac Lab train.py not found at: ${TRAIN_PY}" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

cd "${WORKSPACE_DIR}"
echo "[INFO] Workspace logs: ${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_approach/"
echo "[INFO] Python:         ${PYTHON}"
exec "${PYTHON}" "${TRAIN_PY}" \
  --task Isaac-WidowX-PCB-Approach-v0 \
  "hydra.run.dir=/tmp/widowx_pcb_hydra" \
  "hydra.output_subdir=null" \
  "$@"
