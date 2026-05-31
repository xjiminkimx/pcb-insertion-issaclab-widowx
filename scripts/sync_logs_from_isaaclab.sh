#!/usr/bin/env bash
# Copy rl-games logs from Isaac Lab root into this workspace (one-time migration).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
SRC="${ISAACLAB_ROOT}/logs/rl_games"
DST="${WORKSPACE_DIR}/logs/rl_games"

if [[ ! -d "${SRC}" ]]; then
  echo "[ERROR] No logs at ${SRC}. Train from Isaac Lab root first, or use scripts/train_grasp.sh." >&2
  exit 1
fi

mkdir -p "${DST}"
echo "[INFO] Syncing ${SRC} -> ${DST}"
rsync -a --info=progress2 "${SRC}/" "${DST}/"
echo "[INFO] Done. TensorBoard: python agents/monitor_tensorboard.py"
