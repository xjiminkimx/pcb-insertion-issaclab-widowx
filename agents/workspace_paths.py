"""Paths for WidowX PCB workspace-local training logs."""

from __future__ import annotations

import os

# Package root: .../manager_based/widowx_pcb
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Isaac Lab repo root (five levels above package root).
ISAACLAB_ROOT = os.path.abspath(os.path.join(PACKAGE_ROOT, "..", "..", "..", "..", ".."))

# rl-games writes to <cwd>/logs/rl_games/<name>/ (e.g. widowx_pcb_approach, widowx_pcb_insert).
WORKSPACE_LOGS_DIR = os.path.join(PACKAGE_ROOT, "logs")

APPROACH_EXPERIMENT_NAME = "widowx_pcb_approach"
APPROACH_LOG_DIR = os.path.join(WORKSPACE_LOGS_DIR, "rl_games", APPROACH_EXPERIMENT_NAME)
APPROACH_CHECKPOINT_DIR = os.path.join(APPROACH_LOG_DIR, "nn")
APPROACH_DEFAULT_CHECKPOINT = os.path.join(APPROACH_CHECKPOINT_DIR, f"{APPROACH_EXPERIMENT_NAME}.pth")

INSERT_EXPERIMENT_NAME = "widowx_pcb_insert"
INSERT_LOG_DIR = os.path.join(WORKSPACE_LOGS_DIR, "rl_games", INSERT_EXPERIMENT_NAME)
INSERT_CHECKPOINT_DIR = os.path.join(INSERT_LOG_DIR, "nn")
INSERT_DEFAULT_CHECKPOINT = os.path.join(INSERT_CHECKPOINT_DIR, f"{INSERT_EXPERIMENT_NAME}.pth")

APPROACH_STATES_PATH = os.path.join(PACKAGE_ROOT, "data", "approach_terminal_states.npz")

TRAIN_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "train.py")
PLAY_SCRIPT = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rl_games", "play.py")
