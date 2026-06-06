"""Paths for WidowX PCB workspace-local training logs."""

from __future__ import annotations

import os

# Package root: .../manager_based/widowx_pcb
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Isaac Lab repo root (five levels above package root).
ISAACLAB_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "..", "..", "..", "..", ".."))

# rl-games writes to <cwd>/logs/rl_games/<name>/ (e.g. widowx_pcb_grasp, widowx_pcb_insert).
WORKSPACE_LOGS_DIR = os.path.join(PACKAGE_ROOT, "logs")

TRAIN_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "train.py")
PLAY_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "play.py")
