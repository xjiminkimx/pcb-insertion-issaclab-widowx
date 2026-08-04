# Terminal-state buffers (Sequential Dexterity chaining)

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `approach_terminal_states.npz` | `scripts/collect_approach_states.py` | Insert env reset (`EventCfgInsert`) |
| `push_terminal_states.npz` | *(legacy alias path)* | Same as above if present |
| `insert_terminal_states.npz` | `scripts/collect_insert_states.py` | Insert env reset (if enabled) |

Insert reset: approach buffer robot joints unchanged; gripper held open; PCB XY/Z from buffer;
