"""Paths for WidowX PCB workspace-local training logs."""

from __future__ import annotations

import os

# Package root: .../manager_based/widowx_pcb
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Isaac Lab repo root (five levels above package root).
ISAACLAB_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "..", "..", "..", "..", ".."))

# rl-games writes to <cwd>/logs/rl_games/<name>/ (e.g. widowx_pcb_push, widowx_pcb_slide).
WORKSPACE_LOGS_DIR = os.path.join(PACKAGE_ROOT, "logs")

PUSH_EXPERIMENT_NAME = "widowx_pcb_push"
PUSH_LOG_DIR = os.path.join(WORKSPACE_LOGS_DIR, "rl_games", PUSH_EXPERIMENT_NAME)
PUSH_CHECKPOINT_DIR = os.path.join(PUSH_LOG_DIR, "nn")
PUSH_DEFAULT_CHECKPOINT = os.path.join(PUSH_CHECKPOINT_DIR, f"{PUSH_EXPERIMENT_NAME}.pth")

TRAIN_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "train.py")
PLAY_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "play.py")
