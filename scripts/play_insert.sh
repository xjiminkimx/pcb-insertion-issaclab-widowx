#!/usr/bin/env bash
# Play / evaluate the Insert-phase checkpoint from this workspace's logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
PLAY_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/play.py"

if [[ ! -f "${PLAY_PY}" ]]; then
  echo "[ERROR] Isaac Lab play.py not found at: ${PLAY_PY}" >&2
  exit 1
fi

cd "${WORKSPACE_DIR}"
echo "[INFO] Checkpoint dir: ${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_insert/nn/"
exec python "${PLAY_PY}" --task Isaac-WidowX-PCB-Insert-v0 "$@"
