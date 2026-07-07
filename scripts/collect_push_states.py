#!/usr/bin/env python3
"""Collect successful push terminal states — alias for ``collect_grasp_states.py``."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("collect_grasp_states.py")), run_name="__main__")
