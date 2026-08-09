#!/usr/bin/env bash
# Evaluate success rate for Approach / Insert / Chain policies.
#
# Examples:
#   bash scripts/eval_success.sh approach --num_episodes 200 --num_envs 256 --headless
#   bash scripts/eval_success.sh insert   --num_episodes 200 --num_envs 256 --headless
#   bash scripts/eval_success.sh chain    --num_episodes 50 --headless
#
#   bash scripts/eval_success.sh approach \
#        --checkpoint logs/rl_games/widowx_pcb_approach/weight_saved/widowx_pcb_approach_success_37.pth \
#        --num_episodes 500 --num_envs 512 --headless
#
#   # Run all three sequentially:
#   bash scripts/eval_success.sh all --num_episodes 100 --num_envs 256 --headless
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_PY="${SCRIPT_DIR}/eval_success.py"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

cd "${WORKSPACE_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_success.sh {approach|insert|chain|all} [args...]" >&2
  exit 1
fi

MODE="$1"
shift

echo "[INFO] Python: ${PYTHON}"
echo "[INFO] Mode:   ${MODE}"

run_one() {
  local mode="$1"
  shift
  echo ""
  echo "========== eval_success: ${mode} =========="
  "${PYTHON}" -u "${EVAL_PY}" "${mode}" "$@"
}

case "${MODE}" in
  approach|insert|chain)
    run_one "${MODE}" "$@"
    ;;
  all)
    # Chain ignores --num_envs (forced to 1). Shared args still apply.
    run_one approach "$@"
    run_one insert "$@"
    run_one chain "$@"
    ;;
  *)
    echo "[ERROR] Unknown mode '${MODE}'. Use approach|insert|chain|all." >&2
    exit 1
    ;;
esac
