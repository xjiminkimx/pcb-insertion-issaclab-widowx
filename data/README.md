# Terminal state buffers (*.npz)

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `grasp_terminal_states.npz` | `scripts/collect_grasp_states.py` | Slide env reset (`EventCfgSlide`) |
| `slide_terminal_states.npz` | `scripts/collect_slide_states.py` | Insert env reset (`EventCfgInsert`) |

Each file stores `joint_pos`, `pcb_pos_env`, `pcb_quat`, and `joint_names` at successful phase terminations.

Slide reset: grasp buffer robot joints unchanged; PCB XY/Z from buffer; orientation
flattened to ``_PCB_INIT_ROT_WXYZ`` (flat on conveyor). Tilted buffer rows are skipped.
