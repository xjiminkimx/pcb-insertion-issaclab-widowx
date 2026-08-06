#!/usr/bin/env bash
# Chain-play Approach → Insert with a live straddle handover (one continuous session).
#
# Examples:
#   bash scripts/play_chain.sh --gui
#   bash scripts/play_chain.sh --gui --closedness 0.35 --min_tip_down_deg 12
#   bash scripts/play_chain.sh --headless --no_pitch_gate --episodes 3
#   # Record Approach+Insert as one half-speed (0.5×) mp4:
#   bash scripts/play_chain.sh --video --headless
#   bash scripts/play_chain.sh --gui --debug \
#        --approach_checkpoint logs/rl_games/widowx_pcb_approach/weight_saved/widowx_pcb_approach.pth \
#        --insert_checkpoint logs/rl_games/widowx_pcb_insert/weight_saved/widowx_pcb_insert_05mm_dent.pth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLAY_PY="${SCRIPT_DIR}/play_chain.py"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/isaac-sim/bin/python" ]]; then
  PYTHON="${HOME}/miniconda3/envs/isaac-sim/bin/python"
else
  PYTHON="python3"
fi

cd "${WORKSPACE_DIR}"
echo "[INFO] Python: ${PYTHON}"
echo "[INFO] Chain play: Approach → Insert (live handover)"
exec "${PYTHON}" -u "${PLAY_PY}" "$@"
