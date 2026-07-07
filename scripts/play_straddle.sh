#!/usr/bin/env bash
echo "[WARN] play_straddle.sh was renamed to play_push.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/play_push.sh" "$@"
