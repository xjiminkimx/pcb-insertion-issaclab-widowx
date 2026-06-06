#!/usr/bin/env bash
# Train Insert phase (policy chaining from Grasp).
# Prerequisites:
#   1. Train Grasp phase:    bash scripts/train_grasp.sh --num_envs 2048 --headless
#   2. Collect terminal states:
#        python scripts/collect_grasp_states.py \
#            --checkpoint logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth \
#            --num_envs 256 --num_states 2000 --headless
#   3. Run this script:      bash scripts/train_insert.sh --num_envs 2048 --headless
#
# TensorBoard + checkpoints land in <workspace>/logs/rl_games/widowx_pcb_insert/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
echo "[INFO] Workspace logs: ${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_insert/"
echo "[INFO] Grasp states:   ${GRASP_STATES}"
exec python "${TRAIN_PY}" --task Isaac-WidowX-PCB-Insert-v0 "$@"
