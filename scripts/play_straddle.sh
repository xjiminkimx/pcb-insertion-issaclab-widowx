#!/usr/bin/env bash
# Play / evaluate the Straddle-phase checkpoint from this workspace's logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
PLAY_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/play.py"
CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_straddle/nn"
DEFAULT_CKPT_NAME="widowx_pcb_straddle.pth"

if [[ ! -f "${PLAY_PY}" ]]; then
  echo "[ERROR] Isaac Lab play.py not found at: ${PLAY_PY}" >&2
  exit 1
fi

has_checkpoint=false
use_last=false
for arg in "$@"; do
  if [[ "${arg}" == "--checkpoint" ]]; then
    has_checkpoint=true
  fi
  if [[ "${arg}" == "--use_last_checkpoint" ]]; then
    use_last=true
  fi
done

resolve_checkpoint() {
  local ckpt=""
  if [[ "${use_last}" == true ]]; then
    ckpt="$(ls -t "${CKPT_DIR}"/last_*.pth 2>/dev/null | head -1 || true)"
  elif [[ -f "${CKPT_DIR}/${DEFAULT_CKPT_NAME}" ]]; then
    ckpt="${CKPT_DIR}/${DEFAULT_CKPT_NAME}"
  else
    ckpt="$(ls -t "${CKPT_DIR}"/last_*.pth 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[ERROR] No checkpoint found in: ${CKPT_DIR}" >&2
    echo "        Train first (bash scripts/train_straddle.sh), or pass --checkpoint <path>.pth" >&2
    exit 1
  fi
  echo "${ckpt}"
}

cd "${WORKSPACE_DIR}"

PLAY_ARGS=("$@")
has_num_envs=false
for arg in "${PLAY_ARGS[@]}"; do
  if [[ "${arg}" == "--num_envs" ]]; then
    has_num_envs=true
    break
  fi
done
if [[ "${has_num_envs}" == false ]]; then
  PLAY_ARGS=(--num_envs 4 "${PLAY_ARGS[@]}")
  echo "[INFO] Default --num_envs 4 (compact gap debug prints when num_envs <= 8)"
fi

if [[ "${has_checkpoint}" == false ]]; then
  CKPT="$(resolve_checkpoint)"
  echo "[INFO] Checkpoint: ${CKPT}"
  exec python "${PLAY_PY}" --task Isaac-WidowX-PCB-Straddle-v0 "${PLAY_ARGS[@]}" --checkpoint "${CKPT}"
else
  echo "[INFO] Checkpoint dir: ${CKPT_DIR}/"
  exec python "${PLAY_PY}" --task Isaac-WidowX-PCB-Straddle-v0 "${PLAY_ARGS[@]}"
fi
