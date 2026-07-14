#!/usr/bin/env bash
# Deprecated alias — use play_approach.sh
echo "[WARN] play_push.sh was renamed to play_approach.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/play_approach.sh" "$@"
