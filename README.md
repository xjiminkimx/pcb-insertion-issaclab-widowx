# WidowX PCB insertion (Isaac Lab)

This package is an **Isaac Lab** manager-based reinforcement learning task: a **WidowX** manipulator must keep a **PCB** grasped and **insert it into a target slot** on a magazine, with optional **guide rails** that constrain the approach. The scene, observations, rewards, resets, and terminations are defined in Python using Isaac Lab’s `ManagerBasedRLEnv` pattern and registered as a Gymnasium environment for training with **rl-games** (PPO).

**Task ID:** `Isaac-WidowX-PCB-v0`

**High-level episode flow:**

1. Reset places the PCB in a stable in-gripper grasp (and related geometry).
2. The policy moves the PCB toward a designated slot on the magazine.
3. Rewards encourage alignment and insertion along the configured insertion axis.
4. Terminations end failed episodes early (e.g. drop / fall) and mark success when insertion is stable.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers `Isaac-WidowX-PCB-v0` and applies rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | Scene, actions, observations, rewards, events, terminations, magazine/rail geometry. |
| `mdp_custom.py` | Custom MDP terms (distances, insertion shaping, resets, success/drop checks). |
| `agents/` | PPO config (`rl_games_ppo_cfg.py`), log-std safety helper, TensorBoard monitor script. |
| `jetcobot_assets/` | URDF/USD/meshes used by the task (robot-related assets). |

A local `trossen_ai_isaac/` directory (if present) is intentionally **not** tracked: it is a separate clone with its own `.git`. Track it as a **submodule** or symlink if your workflow depends on it.

---

## Branching model

- **`main`** — release-oriented line; keep this branch in a working, reviewable state.
- **`dev`** — day-to-day integration; merge or rebase into `main` when a change set is ready.

Typical workflow:

```bash
git checkout dev
# edit, commit
git checkout main
git merge dev   # or open a PR from dev → main
```

Both branches are created from the same initial commit when the repository is first set up.

---

## Prerequisites

- [Isaac Sim](https://developer.nvidia.com/isaac-sim) and **Isaac Lab** installed and on your `PYTHONPATH`, consistent with the parent `IsaacLab` repo layout.
- Conda (or equivalent) env that can run Isaac Lab training scripts (example name: `isaac-sim`).

Commands below assume you run training from the **Isaac Lab repository root** (the directory that contains `scripts/`).

---

## Train (rl-games PPO)

```bash
conda activate isaac-sim   # or your env name
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0 --headless
```

Variants:

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0 --video
```

---

## Monitor training (TensorBoard)

From the Isaac Lab repo (adjust path if your checkout differs):

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/widowx_pcb/agents/monitor_tensorboard.py --logdir logs --port 6006
```

Open <http://127.0.0.1:6006> and watch policy/value loss, entropy, KL, and episodic return.

---

## Important task knobs

### Target slot offset

In `widowx_pcb_env_cfg.py`, set the slot center in **magazine local frame**:

```python
TARGET_SLOT_OFFSET = (0.0, 0.0, 0.0)
```

### Insertion axis (magazine frame)

Rewards use `INSERTION_AXIS_LOCAL` (see `widowx_pcb_env_cfg.py`). The default layout uses travel along the magazine **long axis** (local ±X); rotate/sign-flip if your USD layout or slot mouth faces the opposite direction. For top-down insertion, switch to a local ±Z axis and retune `TARGET_SLOT_OFFSET`.

### Physics and collision

If the PCB intersects the magazine visually or behaves unstably: enable/tune CCD and contact offsets in the env config, and ensure the magazine USD has a **collision** representation that matches the real slot opening (concave slots often need careful collision authoring in Omniverse/USD).

### Reset pose and drop termination

Tune gripper-relative PCB spawn parameters in the reset event config and drop thresholds in termination config if you see penetrations at reset or overly long failed episodes.

### Guide rails

Rails form a channel aligned with insertion; constants near `_GUIDE_RAIL_*` and `_MAG_*` in `widowx_pcb_env_cfg.py` control placement and dimensions.

---

## Troubleshooting

- **Body name errors:** Match `SceneEntityCfg("robot", body_names=...)` to link names in your loaded robot USD (e.g. `link_6`, `gripper_left`, `gripper_right`).
- **Penetration at reset:** Slightly increase vertical `pos_offset`, soften edge offsets, or check gripper initial opening.
- **Wrong insertion target:** Usually `TARGET_SLOT_OFFSET` still at the default magazine origin — set it to the true slot center.

---

## Quick workflow

1. Set `TARGET_SLOT_OFFSET` (and insertion axis if needed) for your magazine.
2. Train with the command above.
3. Watch TensorBoard.
4. Adjust reset and termination thresholds if episodes fail too early or never succeed.
