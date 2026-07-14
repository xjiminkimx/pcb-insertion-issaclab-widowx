#!/usr/bin/env bash
echo "[WARN] play_straddle.sh was renamed to play_approach.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/play_approach.sh" "$@"
