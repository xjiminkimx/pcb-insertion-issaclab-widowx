# Terminal state buffers (*.npz)

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `straddle_terminal_states.npz` | `scripts/collect_grasp_states.py` (alias: `collect_straddle_states.py`) | Slide env reset (`EventCfgSlide`) |
| `slide_terminal_states.npz` | `scripts/collect_slide_states.py` | Insert env reset (`EventCfgInsert`) |

Legacy filename `grasp_terminal_states.npz` is still accepted by `train_slide.sh` if present.

Each file stores `joint_pos`, `pcb_pos_env`, `pcb_quat`, and `joint_names` at successful phase terminations.

Slide reset: straddle buffer robot joints unchanged; gripper held open; PCB XY/Z from buffer;
orientation flattened to ``_PCB_INIT_ROT_WXYZ`` (flat on conveyor). Tilted buffer rows are skipped.
