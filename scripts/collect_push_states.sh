#!/usr/bin/env bash
# Deprecated alias — use collect_approach_states.sh
echo "[WARN] collect_push_states.sh was renamed to collect_approach_states.sh" >&2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/collect_approach_states.sh" "$@"
