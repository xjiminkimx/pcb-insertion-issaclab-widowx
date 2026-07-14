# Terminal-state buffers (Sequential Dexterity chaining)

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `approach_terminal_states.npz` | `scripts/collect_approach_states.py` | Slide env reset (`EventCfgSlide`) |
| `push_terminal_states.npz` | *(legacy alias path)* | Same as above if present |
| `slide_terminal_states.npz` | `scripts/collect_slide_states.py` | Insert env reset (if enabled) |

Slide reset: approach buffer robot joints unchanged; gripper held open; PCB XY/Z from buffer;
