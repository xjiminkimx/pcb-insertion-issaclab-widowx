#!/usr/bin/env bash
echo "[WARN] collect_straddle_states.sh was renamed to collect_push_states.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/collect_push_states.sh" "$@"
