#!/usr/bin/env bash
# Train Slide phase (policy chaining from Approach).
# Prerequisites:
#   1. Train Approach: bash scripts/train_approach.sh --num_envs 2048 --headless
#   2. Collect approach terminal states: bash scripts/collect_approach_states.sh
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
  if [[ -d "${SLIDE_LOG_DIR}" ]]; then
    echo "[INFO] Removing previous slide logs: ${SLIDE_LOG_DIR}"
    rm -rf "${SLIDE_LOG_DIR}"
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

APPROACH_STATES="${WORKSPACE_DIR}/data/approach_terminal_states.npz"
LEGACY_PUSH_STATES="${WORKSPACE_DIR}/data/push_terminal_states.npz"
LEGACY_STRADDLE_STATES="${WORKSPACE_DIR}/data/straddle_terminal_states.npz"
LEGACY_GRASP_STATES="${WORKSPACE_DIR}/data/grasp_terminal_states.npz"
if [[ -f "${APPROACH_STATES}" ]]; then
  STATES_PATH="${APPROACH_STATES}"
elif [[ -f "${LEGACY_PUSH_STATES}" ]]; then
  echo "[WARN] Using legacy buffer: ${LEGACY_PUSH_STATES} (re-collect with collect_approach_states.py)" >&2
  STATES_PATH="${LEGACY_PUSH_STATES}"
elif [[ -f "${LEGACY_STRADDLE_STATES}" ]]; then
  echo "[WARN] Using legacy buffer: ${LEGACY_STRADDLE_STATES} (re-collect with collect_approach_states.py)" >&2
  STATES_PATH="${LEGACY_STRADDLE_STATES}"
elif [[ -f "${LEGACY_GRASP_STATES}" ]]; then
  echo "[WARN] Using legacy buffer: ${LEGACY_GRASP_STATES}" >&2
  STATES_PATH="${LEGACY_GRASP_STATES}"
else
  echo "[ERROR] Approach terminal-state buffer not found: ${APPROACH_STATES}" >&2
  echo "        Run: python scripts/collect_approach_states.py --checkpoint <approach.pth>" >&2
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
echo "[INFO] Approach states: ${STATES_PATH}"
echo "[INFO] Python:         ${PYTHON}"

"${PYTHON}" "${TRAIN_PY}" \
  --task Isaac-WidowX-PCB-Slide-v0 \
  "hydra.run.dir=/tmp/widowx_pcb_hydra" \
  "hydra.output_subdir=null" \
  "${TRAIN_ARGS[@]}"
