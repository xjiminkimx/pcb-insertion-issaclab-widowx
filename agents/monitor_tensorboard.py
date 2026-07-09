#!/usr/bin/env python3
"""Launch TensorBoard for WidowX PCB training logs.

Usage:
    python3 monitor_tensorboard.py
    python3 monitor_tensorboard.py --logdir /path/to/IsaacLab/logs --port 6006

This helper script avoids typing long TensorBoard commands repeatedly and
prints WidowX-specific run paths plus useful scalar tags to monitor.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

try:
    from .workspace_paths import ISAACLAB_ROOT, WORKSPACE_LOGS_DIR
except ImportError:
    from workspace_paths import ISAACLAB_ROOT, WORKSPACE_LOGS_DIR

# Known rl-games experiment layout for this task (relative to --logdir/rl_games/).
WIDOWX_RL_RUNS = (
    ("Push", "widowx_pcb_push/summaries"),
    ("Grasp", "widowx_pcb_grasp/summaries"),
    ("Grasp gripper test", "widowx_pcb_grasp_gripper_test/summaries"),
    ("Slide", "widowx_pcb_slide/summaries"),
    ("Insert", "widowx_pcb_insert/summaries"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch TensorBoard for rl-games logs.")
    parser.add_argument(
        "--logdir",
        type=str,
        default=WORKSPACE_LOGS_DIR,
        help=f"Logs root (default: workspace logs at {WORKSPACE_LOGS_DIR}).",
    )
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard HTTP port.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host (use 0.0.0.0 for remote access).",
    )
    return parser.parse_args()


def _find_existing_logdir(user_logdir: str) -> str | None:
    """Resolve a usable log directory from current workspace layout.

    Priority:
    1) user-provided path as-is (absolute or relative to cwd)
    2) workspace logs (``WORKSPACE_LOGS_DIR``)
    3) Isaac Lab root logs (``<IsaacLab>/logs``)
    4) walk upward from cwd and try "<ancestor>/<user_logdir>" and "<ancestor>/logs"
    """
    candidates: list[str] = []
    if os.path.isabs(user_logdir):
        candidates.append(user_logdir)
    else:
        candidates.append(os.path.abspath(user_logdir))

    candidates.append(WORKSPACE_LOGS_DIR)
    candidates.append(os.path.join(ISAACLAB_ROOT, "logs"))

    cwd = os.getcwd()
    current = cwd
    while True:
        candidates.append(os.path.join(current, user_logdir))
        candidates.append(os.path.join(current, "logs"))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    seen: set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            return os.path.abspath(path)
    return None


def _is_widowx_package_cwd() -> bool:
    return os.path.basename(os.getcwd()) == "widowx_pcb"


def _find_widowx_summary_dirs(logdir: str) -> list[tuple[str, str]]:
    """Return (phase_label, absolute_summaries_path) for runs that exist on disk."""
    found: list[tuple[str, str]] = []
    for phase, rel in WIDOWX_RL_RUNS:
        summaries = os.path.join(logdir, "rl_games", rel)
        if os.path.isdir(summaries):
            found.append((phase, summaries))
    return found


def _print_cwd_hint() -> None:
    if not _is_widowx_package_cwd():
        return
    print("[HINT] Train with scripts/train_push.sh / train_slide.sh / train_insert.sh")
    print("       so logs land in ./logs/rl_games/ (this workspace).")
    print("       To copy existing Isaac Lab logs: bash scripts/sync_logs_from_isaaclab.sh")
    print()


def _print_run_paths(logdir: str) -> None:
    found = _find_widowx_summary_dirs(logdir)
    print("[INFO] WidowX PCB run folders (relative to logdir):")
    for phase, rel in WIDOWX_RL_RUNS:
        summaries = os.path.join(logdir, "rl_games", rel)
        if os.path.isdir(summaries):
            status = "found"
            n_events = len([f for f in os.listdir(summaries) if f.startswith("events.out.tfevents")])
            extra = f" ({n_events} event file{'s' if n_events != 1 else ''})"
        else:
            status = "not found (train this phase first)"
            extra = ""
        print(f"       {phase:5}  rl_games/{rel}  [{status}]{extra}")
    print()
    if not found:
        print("[WARN] No WidowX summary folders under this logdir yet.")
        print("       Start training, then re-run this script.")
        print()


def main() -> int:
    args = parse_args()
    _print_cwd_hint()
    logdir = _find_existing_logdir(args.logdir)

    if logdir is None:
        print(f"[ERROR] Could not find a valid log directory for: {args.logdir}")
        print("Run training first or pass an explicit --logdir")
        print("(example: --logdir /home/<user>/Documents/IsaacLab/logs).")
        return 1

    print(f"[INFO] Launching TensorBoard")
    print(f"       logdir: {logdir}")
    print(f"       url:    http://{args.host}:{args.port}")
    print()
    _print_run_paths(logdir)
    print("[TIP] Useful scalars to watch in the TensorBoard UI:")
    print("      - episodic reward (rl-games rewards tag)")
    print("      - policy loss / actor loss")
    print("      - value loss / critic loss")
    print("      - entropy")
    print("      - KL / approx_kl")
    print()
    print("[TIP] Run folders under logs/rl_games/:")
    print("      - widowx_pcb_push (open-jaw approach + +Y slide)")
    print("      - widowx_pcb_grasp    (legacy grasp phase)")
    print("      - widowx_pcb_slide    (phase 2 — slide to slot mouth)")
    print("      - widowx_pcb_insert   (phase 3 — SDF insert)")
    print()
    print("[TIP] Push monitoring (Curriculum/push_gripper_debug/*):")
    print("      - closedness_mean / closedness_live  — proximity σ (same as finger_proximity reward)")
    print("      - closedness_ep_max                — best env per episode (rises before batch mean)")
    print("      - closedness_tight_mean            — tight 20 mm σ (success placement)")
    print("      - dist_l_mm_mean / dist_r_mm_mean   — pad→±20 mm target distance")
    print("      - push_gate_open_frac / push_gate_open_live")
    print("      - travel_frac_ep_max / travel_frac_end     — slide +Y progress (0→1)")
    print("      - milestone_bonus_ep                       — tiers hit per episode (0–5)")
    print("      - milestone_tier_*_hit_frac                — per-tier hit rate")
    print("      - milestone_pose_ok_frac                   — in-lane X/Z while sliding")
    print()

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        logdir,
        "--port",
        str(args.port),
        "--host",
        args.host,
    ]

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n[INFO] TensorBoard stopped by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
