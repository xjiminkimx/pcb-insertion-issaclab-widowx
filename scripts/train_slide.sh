#!/usr/bin/env bash
# Train Slide phase (policy chaining from Grasp).
# Prerequisites:
#   1. Train Grasp: bash scripts/train_grasp.sh --num_envs 2048 --headless
#   2. Collect grasp terminal states (see scripts/collect_grasp_states.py)
#   3. Run this script
#
# Max epochs: agents/rl_games_ppo_cfg.py → WidowXPcbSlidePPOCfg max_epochs
# TensorBoard + checkpoints: logs/rl_games/widowx_pcb_slide/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_ARGS=()
CLEAN_TENSORBOARD=0
CLEAN_ALL=0
for arg in "$@"; do
  if [[ "${arg}" == "--clean-logs" ]]; then
    CLEAN_TENSORBOARD=1
  elif [[ "${arg}" == "--clean-all" ]]; then
    CLEAN_ALL=1
  else
    TRAIN_ARGS+=("${arg}")
  fi
done

SLIDE_LOG_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_slide"
if [[ "${CLEAN_ALL}" -eq 1 ]]; then
  OUTPUTS_DIR="${WORKSPACE_DIR}/outputs"
  if [[ -d "${SLIDE_LOG_DIR}" ]]; then
    echo "[INFO] Removing previous slide logs: ${SLIDE_LOG_DIR}"
    rm -rf "${SLIDE_LOG_DIR}"
  fi
  if [[ -d "${OUTPUTS_DIR}" ]]; then
    echo "[INFO] Removing Hydra outputs: ${OUTPUTS_DIR}"
    rm -rf "${OUTPUTS_DIR}"
  fi
elif [[ "${CLEAN_TENSORBOARD}" -eq 1 ]]; then
  SUMMARIES_DIR="${SLIDE_LOG_DIR}/summaries"
  if [[ -d "${SUMMARIES_DIR}" ]]; then
    echo "[INFO] Clearing TensorBoard summaries (checkpoints kept): ${SUMMARIES_DIR}"
    rm -rf "${SUMMARIES_DIR}"
  fi
fi

ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
TRAIN_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/train.py"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[ERROR] Isaac Lab train.py not found at: ${TRAIN_PY}" >&2
  exit 1
fi

GRASP_STATES="${WORKSPACE_DIR}/data/grasp_terminal_states.npz"
if [[ ! -f "${GRASP_STATES}" ]]; then
  echo "[ERROR] Grasp terminal-state buffer not found: ${GRASP_STATES}" >&2
  echo "        Run: python scripts/collect_grasp_states.py --checkpoint <grasp.pth>" >&2
  exit 1
fi

cd "${WORKSPACE_DIR}"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

echo "[INFO] Workspace logs: ${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_slide/"
echo "[INFO] Grasp states:   ${GRASP_STATES}"
echo "[INFO] Python:         ${PYTHON}"

"${PYTHON}" "${TRAIN_PY}" \
  --task Isaac-WidowX-PCB-Slide-v0 \
  "${TRAIN_ARGS[@]}"
