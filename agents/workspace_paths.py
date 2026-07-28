"""Paths for WidowX PCB workspace-local training logs."""

from __future__ import annotations

import os

# Package root: .../manager_based/widowx_pcb
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Isaac Lab repo root (five levels above package root).
ISAACLAB_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "..", "..", "..", "..", ".."))

# rl-games writes to <cwd>/logs/rl_games/<name>/ (e.g. widowx_pcb_approach, widowx_pcb_slide).
WORKSPACE_LOGS_DIR = os.path.join(PACKAGE_ROOT, "logs")

APPROACH_EXPERIMENT_NAME = "widowx_pcb_approach"
APPROACH_LOG_DIR = os.path.join(WORKSPACE_LOGS_DIR, "rl_games", APPROACH_EXPERIMENT_NAME)
APPROACH_CHECKPOINT_DIR = os.path.join(APPROACH_LOG_DIR, "nn")
APPROACH_DEFAULT_CHECKPOINT = os.path.join(APPROACH_CHECKPOINT_DIR, f"{APPROACH_EXPERIMENT_NAME}.pth")

SLIDE_EXPERIMENT_NAME = "widowx_pcb_slide"
SLIDE_LOG_DIR = os.path.join(WORKSPACE_LOGS_DIR, "rl_games", SLIDE_EXPERIMENT_NAME)
SLIDE_CHECKPOINT_DIR = os.path.join(SLIDE_LOG_DIR, "nn")
SLIDE_DEFAULT_CHECKPOINT = os.path.join(SLIDE_CHECKPOINT_DIR, f"{SLIDE_EXPERIMENT_NAME}.pth")

APPROACH_STATES_PATH = os.path.join(PACKAGE_ROOT, "data", "approach_terminal_states.npz")

TRAIN_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "train.py")
PLAY_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "play.py")

# Deprecated aliases (pre-approach rename).
PUSH_EXPERIMENT_NAME = APPROACH_EXPERIMENT_NAME
PUSH_LOG_DIR = APPROACH_LOG_DIR
PUSH_CHECKPOINT_DIR = APPROACH_CHECKPOINT_DIR
PUSH_DEFAULT_CHECKPOINT = APPROACH_DEFAULT_CHECKPOINT
PUSH_STATES_PATH = APPROACH_STATES_PATH
