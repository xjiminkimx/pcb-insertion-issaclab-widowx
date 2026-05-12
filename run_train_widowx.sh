#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run_train_widowx.sh [--env ENV_NAME] [--no-conda] [train.py args...]

Convenience launcher for the WidowX PCB task. It finds the IsaacLab repo root,
optionally activates a Conda environment, and forwards any extra arguments to
rl-games train.py.

Options:
  --env ENV_NAME  Conda environment to activate (default: isaac-sim)
  --no-conda      Skip Conda activation and use the current shell environment
  -h, --help      Show this help message

Examples:
  ./run_train_widowx.sh
  ./run_train_widowx.sh --env isaac-sim --num_envs 512 --seed 0
  ./run_train_widowx.sh --no-conda --headless
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-isaac-sim}"
USE_CONDA=1
TRAIN_ARGS=()

while (($#)); do
  case "$1" in
    --env)
      if [[ $# -lt 2 ]]; then
        echo "Error: --env requires a value." >&2
        exit 1
      fi
      CONDA_ENV_NAME="$2"
      shift 2
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

find_isaaclab_root() {
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/scripts/reinforcement_learning/rl_games/train.py" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

activate_conda_env() {
  if [[ "$USE_CONDA" -eq 0 ]]; then
    return 0
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available in PATH." >&2
    echo "Activate your environment manually first or rerun with --no-conda." >&2
    exit 1
  fi

  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    echo "Error: could not locate conda.sh for environment activation." >&2
    echo "Activate your environment manually first or rerun with --no-conda." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
  echo "Activated conda environment: ${CONDA_ENV_NAME}"
}

ISAACLAB_ROOT="$(find_isaaclab_root)" || {
  echo "Error: could not locate IsaacLab repo root from ${SCRIPT_DIR}." >&2
  exit 1
}

activate_conda_env

cd "$ISAACLAB_ROOT"
echo "IsaacLab root: ${ISAACLAB_ROOT}"

python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-WidowX-PCB-v0 \
  "${TRAIN_ARGS[@]}"
