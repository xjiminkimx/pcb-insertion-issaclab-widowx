#!/usr/bin/env bash
# Train Insert phase (policy chaining from Grasp).
# Prerequisites:
#   1. Train Grasp phase:    bash scripts/train_grasp.sh --num_envs 2048 --headless
#   2. Collect terminal states:
#        python scripts/collect_grasp_states.py \
#            --checkpoint logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth \
#            --num_envs 256 --num_states 2000 --headless
#   3. Run this script:      bash scripts/train_insert.sh --num_envs 2048 --headless
#      Clear TensorBoard only: add --clean-logs  (keeps checkpoints in nn/)
#      Wipe everything:        add --clean-all   (nn/, summaries/, videos/, outputs/)
#
# Max epochs: agents/rl_games_ppo_cfg.py → WidowXPcbInsertPPOCfg max_epochs
#            (override at runtime: --max_iterations N)
# TensorBoard + checkpoints land in <workspace>/logs/rl_games/widowx_pcb_insert/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional log cleanup (not passed to train.py).
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

INSERT_LOG_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_insert"
if [[ "${CLEAN_ALL}" -eq 1 ]]; then
  OUTPUTS_DIR="${WORKSPACE_DIR}/outputs"
  if [[ -d "${INSERT_LOG_DIR}" ]]; then
    echo "[INFO] Removing all insert logs: ${INSERT_LOG_DIR}"
    rm -rf "${INSERT_LOG_DIR}"
  fi
  if [[ -d "${OUTPUTS_DIR}" ]]; then
    echo "[INFO] Removing Hydra outputs: ${OUTPUTS_DIR}"
    rm -rf "${OUTPUTS_DIR}"
  fi
elif [[ "${CLEAN_TENSORBOARD}" -eq 1 ]]; then
  SUMMARIES_DIR="${INSERT_LOG_DIR}/summaries"
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
  echo "        Run: python scripts/collect_grasp_states.py --checkpoint <path>.pth" >&2
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

echo "[INFO] Workspace logs: ${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_insert/"
echo "[INFO] Grasp states:   ${GRASP_STATES}"
echo "[INFO] Python:         ${PYTHON}"

"${PYTHON}" "${TRAIN_PY}" \
  --task Isaac-WidowX-PCB-Insert-v0 \
  "${TRAIN_ARGS[@]}"
