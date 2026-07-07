# Terminal state buffers (*.npz)

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `push_terminal_states.npz` | `scripts/collect_grasp_states.py` (alias: `collect_push_states.py`) | Slide env reset (`EventCfgSlide`) |
| `slide_terminal_states.npz` | `scripts/collect_slide_states.py` | Insert env reset (`EventCfgInsert`) |

Legacy filenames `straddle_terminal_states.npz` and `grasp_terminal_states.npz` are still accepted by `train_slide.sh` if present.

Each file stores `joint_pos`, `pcb_pos_env`, `pcb_quat`, and `joint_names` at successful task terminations.

Slide reset: push buffer robot joints unchanged; gripper held open; PCB XY/Z from buffer;
orientation flattened to ``_PCB_INIT_ROT_WXYZ`` (flat on conveyor). Tilted buffer rows are skipped.
