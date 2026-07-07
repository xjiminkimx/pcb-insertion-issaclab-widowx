#!/usr/bin/env bash
echo "[WARN] train_straddle.sh was renamed to train_push.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_push.sh" "$@"
