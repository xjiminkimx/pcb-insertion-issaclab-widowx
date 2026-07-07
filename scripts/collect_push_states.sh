#!/usr/bin/env bash
# Collect push-task terminal states for downstream training buffers.
# Defaults (in collect_grasp_states.py): num_envs=4096, num_states=500, headless.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/collect_grasp_states.sh" "$@"
