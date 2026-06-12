# Terminal state buffers (*.npz)

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `grasp_terminal_states.npz` | `scripts/collect_grasp_states.py` | Slide env reset (`EventCfgSlide`) |
| `slide_terminal_states.npz` | `scripts/collect_slide_states.py` | Insert env reset (`EventCfgInsert`) |

Each file stores `joint_pos`, `pcb_pos_env`, `pcb_quat`, and `joint_names` at successful phase terminations.
