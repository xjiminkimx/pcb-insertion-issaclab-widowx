#!/usr/bin/env bash
# Train Approach phase; TensorBoard + checkpoints in logs/rl_games/widowx_pcb_approach/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
TRAIN_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/train.py"
CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_approach/nn"

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

resolve_resume_checkpoint() {
  local best="${CKPT_DIR}/widowx_pcb_approach.pth"
  local ckpt
  if [[ -f "${best}" ]]; then
    ckpt="${best}"
    echo "[INFO] Using best checkpoint: ${ckpt}" >&2
  else
    ckpt="$(ls -t "${CKPT_DIR}"/last_*.pth 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[ERROR] No checkpoint found in: ${CKPT_DIR}" >&2
    echo "        Train from scratch first, or pass --checkpoint <path>.pth" >&2
    exit 1
  fi
  echo "${ckpt}"
}

checkpoint_epoch() {
  "${PYTHON}" - "${1}" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("epoch", 0)))
PY
}

patch_approach_checkpoint() {
  "${PYTHON}" - "${1}" <<'PY'
import sys
from agents.checkpoint_compat import ensure_approach_checkpoint_compatible

print(ensure_approach_checkpoint_compatible(sys.argv[1]))
PY
}

TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      EXTRA_EPOCHS=100
      shift
      if [[ $# -gt 0 && "${1}" =~ ^[0-9]+$ ]]; then
        EXTRA_EPOCHS="$1"
        shift
      fi
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${RESUME:-0}" -eq 1 ]]; then
  has_checkpoint=false
  has_max_iterations=false
  for arg in "${TRAIN_ARGS[@]}"; do
    if [[ "${arg}" == "--checkpoint" ]]; then
      has_checkpoint=true
    fi
    if [[ "${arg}" == "--max_iterations" ]]; then
      has_max_iterations=true
    fi
  done

  if [[ "${has_checkpoint}" == false ]]; then
    CKPT="$(resolve_resume_checkpoint)"
    CKPT="$(patch_approach_checkpoint "${CKPT}")"
    TRAIN_ARGS=(--checkpoint "${CKPT}" "${TRAIN_ARGS[@]}")
    echo "[INFO] Resume checkpoint: ${CKPT}"
  fi

  if [[ "${has_max_iterations}" == false ]]; then
    resume_ckpt=""
    i=0
    while [[ $i -lt ${#TRAIN_ARGS[@]} ]]; do
      if [[ "${TRAIN_ARGS[$i]}" == "--checkpoint" ]]; then
        ((i++)) || true
        resume_ckpt="${TRAIN_ARGS[$i]}"
        break
      fi
      ((i++)) || true
    done
    if [[ -z "${resume_ckpt}" ]]; then
      echo "[ERROR] --resume requires a checkpoint path" >&2
      exit 1
    fi
    START_EPOCH="$(checkpoint_epoch "${resume_ckpt}")"
    MAX_EPOCHS="$((START_EPOCH + EXTRA_EPOCHS))"
    TRAIN_ARGS=(--max_iterations "${MAX_EPOCHS}" "${TRAIN_ARGS[@]}")
    echo "[INFO] Resume from epoch ${START_EPOCH}, training to epoch ${MAX_EPOCHS} (+${EXTRA_EPOCHS})"
  fi
fi

# Isaac Lab train.py only resamples when CLI ``--seed -1``; agent cfg alone is not enough.
has_seed=false
for arg in "${TRAIN_ARGS[@]}"; do
  if [[ "${arg}" == "--seed" ]]; then
    has_seed=true
    break
  fi
done
if [[ "${has_seed}" == false ]]; then
  TRAIN_ARGS=(--seed -1 "${TRAIN_ARGS[@]}")
fi

cd "${WORKSPACE_DIR}"
echo "[INFO] Workspace logs: ${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_approach/"
echo "[INFO] Python:         ${PYTHON}"
exec "${PYTHON}" "${TRAIN_PY}" \
  --task Isaac-WidowX-PCB-Approach-v0 \
  "hydra.run.dir=/tmp/widowx_pcb_hydra" \
  "hydra.output_subdir=null" \
  "${TRAIN_ARGS[@]}"
