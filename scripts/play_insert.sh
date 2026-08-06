#!/usr/bin/env bash
# Play / evaluate the Insert-phase checkpoint from this workspace's logs/.
# Default: 16 parallel envs (override with --num_envs N).
#
# Debug leading-edge vs insert_success box (console):
#   bash scripts/play_insert.sh --debug --num_envs 4 --headless
#   bash scripts/play_insert.sh --debug --debug-env 0 --debug-every 16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="$(cd "${WORKSPACE_DIR}/../../../../.." && pwd)"
PLAY_PY="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/play.py"
CKPT_DIR="${WORKSPACE_DIR}/logs/rl_games/widowx_pcb_insert/nn"
DEFAULT_CKPT_NAME="widowx_pcb_insert.pth"
NUM_ENVS=16

if [[ ! -f "${PLAY_PY}" ]]; then
  echo "[ERROR] Isaac Lab play.py not found at: ${PLAY_PY}" >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

has_checkpoint=false
use_last=false
has_num_envs=false
debug=false
debug_env=0
debug_every=32
PLAY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      debug=true
      shift
      ;;
    --debug-env)
      debug_env="${2:?--debug-env requires an integer env id}"
      shift 2
      ;;
    --debug-every)
      debug_every="${2:?--debug-every requires a step interval}"
      shift 2
      ;;
    *)
      if [[ "$1" == "--checkpoint" ]]; then
        has_checkpoint=true
      fi
      if [[ "$1" == "--use_last_checkpoint" ]]; then
        use_last=true
      fi
      if [[ "$1" == "--num_envs" ]]; then
        has_num_envs=true
      fi
      PLAY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${debug}" == true && "${has_num_envs}" == false ]]; then
  NUM_ENVS=4
  PLAY_ARGS=(--num_envs "${NUM_ENVS}" "${PLAY_ARGS[@]}")
  echo "[INFO] --debug: default --num_envs ${NUM_ENVS} (compact console prints when num_envs <= 8)"
elif [[ "${has_num_envs}" == false ]]; then
  PLAY_ARGS=(--num_envs "${NUM_ENVS}" "${PLAY_ARGS[@]}")
fi

HYDRA_DEBUG=()
if [[ "${debug}" == true ]]; then
  HYDRA_DEBUG=(
    "env.events.insert_success_debug.params.enable_print=true"
    "env.events.insert_success_debug.params.print_env_id=${debug_env}"
    "env.events.insert_success_debug.params.print_every_control_steps=${debug_every}"
  )
  echo "[INFO] Insert success debug: env=${debug_env}, every ${debug_every} control steps"
fi

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
    echo "        Train first, or pass --checkpoint <path>.pth" >&2
    exit 1
  fi
  echo "${ckpt}"
}

cd "${WORKSPACE_DIR}"

# Isaac Lab play.py only resamples when CLI ``--seed -1``.
has_seed=false
for arg in "${PLAY_ARGS[@]}"; do
  if [[ "${arg}" == "--seed" ]]; then
    has_seed=true
    break
  fi
done
if [[ "${has_seed}" == false ]]; then
  PLAY_ARGS=(--seed -1 "${PLAY_ARGS[@]}")
fi

echo "[INFO] Python: ${PYTHON}"
if [[ "${has_checkpoint}" == false ]]; then
  CKPT="$(resolve_checkpoint)"
  echo "[INFO] Checkpoint: ${CKPT}"
  exec "${PYTHON}" "${PLAY_PY}" \
    --task Isaac-WidowX-PCB-Insert-v0 \
    "hydra.run.dir=/tmp/widowx_pcb_hydra" \
    "hydra.output_subdir=null" \
    "${HYDRA_DEBUG[@]}" \
    "${PLAY_ARGS[@]}" \
    --checkpoint "${CKPT}"
else
  echo "[INFO] Checkpoint dir: ${CKPT_DIR}/"
  exec "${PYTHON}" "${PLAY_PY}" \
    --task Isaac-WidowX-PCB-Insert-v0 \
    "hydra.run.dir=/tmp/widowx_pcb_hydra" \
    "hydra.output_subdir=null" \
    "${HYDRA_DEBUG[@]}" \
    "${PLAY_ARGS[@]}"
fi
