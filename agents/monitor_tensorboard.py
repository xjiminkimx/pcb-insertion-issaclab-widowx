#!/usr/bin/env python3
"""Launch TensorBoard for WidowX PCB training logs.

Usage:
    python3 monitor_tensorboard.py
    python3 monitor_tensorboard.py --logdir /path/to/logs --port 6006

This helper script avoids typing long TensorBoard commands repeatedly and
prints a short checklist of useful loss/diagnostic tags to monitor.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch TensorBoard for rl-games logs.")
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Root directory that contains IsaacLab/rl-games training logs.",
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
    1) user-provided path as-is
    2) user-provided path relative to cwd
    3) nearest ancestor directory that contains a `logs/` folder
    """
    candidates = []
    if os.path.isabs(user_logdir):
        candidates.append(user_logdir)
    else:
        candidates.append(os.path.abspath(user_logdir))

    # Walk upward from cwd and try "<ancestor>/<user_logdir>" and "<ancestor>/logs".
    cwd = os.getcwd()
    current = cwd
    while True:
        candidates.append(os.path.join(current, user_logdir))
        candidates.append(os.path.join(current, "logs"))
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return None


def main() -> int:
    args = parse_args()
    logdir = _find_existing_logdir(args.logdir)

    if logdir is None:
        print(f"[ERROR] Could not find a valid log directory for: {args.logdir}")
        print("Run training first or pass an explicit --logdir (example: --logdir /home/<user>/Documents/IsaacLab/logs).")
        return 1

    print(f"[INFO] Launching TensorBoard")
    print(f"       logdir: {logdir}")
    print(f"       url:    http://{args.host}:{args.port}")
    print()
    print("[TIP] Useful scalars to watch:")
    print("      - policy loss / actor loss")
    print("      - value loss / critic loss")
    print("      - entropy")
    print("      - KL / approx_kl")
    print("      - episodic reward")
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
