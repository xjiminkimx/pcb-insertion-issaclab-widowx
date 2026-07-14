#!/usr/bin/env bash
# Deprecated alias — use train_approach.sh
echo "[WARN] train_push.sh was renamed to train_approach.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_approach.sh" "$@"
