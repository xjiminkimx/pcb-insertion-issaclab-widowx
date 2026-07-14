#!/usr/bin/env bash
echo "[WARN] train_straddle.sh was renamed to train_approach.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_approach.sh" "$@"
