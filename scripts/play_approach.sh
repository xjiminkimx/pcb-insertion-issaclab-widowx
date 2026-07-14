#!/usr/bin/env bash
# Play / evaluate the Approach-task checkpoint from this workspace's logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
PLAY_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/play.py"
CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_approach/nn"
LEGACY_CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_push/nn"
LEGACY_STRADDLE_CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_straddle/nn"
DEFAULT_CKPT_NAME="widowx_pcb_approach.pth"
LEGACY_CKPT_NAME="widowx_pcb_push.pth"
LEGACY_STRADDLE_CKPT_NAME="widowx_pcb_straddle.pth"

if [[ ! -d "${CKPT_DIR}" && -d "${LEGACY_CKPT_DIR}" ]]; then
  echo "[WARN] Using legacy log dir: ${LEGACY_CKPT_DIR} (migrate to widowx_pcb_approach)" >&2
  CKPT_DIR="${LEGACY_CKPT_DIR}"
  DEFAULT_CKPT_NAME="${LEGACY_CKPT_NAME}"
elif [[ ! -d "${CKPT_DIR}" && -d "${LEGACY_STRADDLE_CKPT_DIR}" ]]; then
  echo "[WARN] Using legacy log dir: ${LEGACY_STRADDLE_CKPT_DIR}" >&2
  CKPT_DIR="${LEGACY_STRADDLE_CKPT_DIR}"
  DEFAULT_CKPT_NAME="${LEGACY_STRADDLE_CKPT_NAME}"
fi

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
    echo "        Train first (bash scripts/train_approach.sh), or pass --checkpoint <path>.pth" >&2
    exit 1
  fi
  echo "${ckpt}"
}

patch_approach_checkpoint() {
  python - "${1}" <<'PY'
import sys
from agents.checkpoint_compat import ensure_approach_checkpoint_compatible

print(ensure_approach_checkpoint_compatible(sys.argv[1]))
PY
}

patch_checkpoint_args() {
  local -n args_ref=$1
  local patched=()
  local i=0
  while [[ $i -lt ${#args_ref[@]} ]]; do
    local arg="${args_ref[$i]}"
    if [[ "${arg}" == "--checkpoint" ]]; then
      patched+=("--checkpoint")
      ((i++)) || true
      local raw="${args_ref[$i]:-}"
      if [[ -z "${raw}" ]]; then
        echo "[ERROR] --checkpoint requires a path" >&2
        exit 1
      fi
      local ckpt
      ckpt="$(patch_approach_checkpoint "${raw}")"
      if [[ "${ckpt}" != "${raw}" ]]; then
        echo "[INFO] Patched checkpoint for 44-dim obs: ${ckpt}" >&2
      fi
      patched+=("${ckpt}")
    else
      patched+=("${arg}")
    fi
    ((i++)) || true
  done
  args_ref=("${patched[@]}")
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
  CKPT="$(patch_approach_checkpoint "${CKPT}")"
  echo "[INFO] Checkpoint: ${CKPT}"
  PLAY_ARGS=(--checkpoint "${CKPT}" "${PLAY_ARGS[@]}")
else
  patch_checkpoint_args PLAY_ARGS
fi

echo "[INFO] Checkpoint dir: ${CKPT_DIR}/"
exec python "${PLAY_PY}" --task Isaac-WidowX-PCB-Approach-v0 "${PLAY_ARGS[@]}"
