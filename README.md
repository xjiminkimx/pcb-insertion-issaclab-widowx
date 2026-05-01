# WidowX PCB on-rail task (Isaac Lab)

This package is an **Isaac Lab** manager-based RL task: a **WidowX** arm interacts with a **PCB** on a **magazine + guide rail** assembly (single aligned USD). **Target-slot distance rewards and slot observations have been removed** so you can add forward-push / rail objectives separately. Training uses **rl-games** (PPO).

**Task ID:** `Isaac-WidowX-PCB-v0`

**High-level episode flow:**

1. Reset places the PCB on the rail pose (see `_PCB_INIT_*` and reset events).
2. Dense rewards are regularization-style (height, floor, tilt, etc.) — no slot target.
3. Terminations: PCB too low / dropped; success termination is left for you to define with the new task reward.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers `Isaac-WidowX-PCB-v0` and applies rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | Scene, actions, observations, rewards, events, terminations, magazine/rail geometry. |
| `mdp_custom.py` | Custom MDP terms (regularization, rail reset, drop detection; optional in-gripper reset). |
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

### Fixture pose vs USD

Align `_MAG_POS` / `_MAG_ROT_WXYZ` in `widowx_pcb_env_cfg.py` with your imported **magazine + rail** asset in world frame. Analytic `_GUIDE_RAIL_*` may not match `usd_model/magazine.usd` collision—if the **green PCB clips into white rails**, raise **`_PCB_SPAWN_Z_BIAS`** (and optionally `collision_props.contact_offset` / `rest_offset` on PCB + magazine) until the board sits on the rail tops in the contact view.

### Physics and collision

If the PCB intersects the fixture: tune CCD and contact offsets in the env config and collision meshes in USD.

### Reset pose and drop termination

Tune **`pcb_tilt_excessive`**, **`pcb_long_axis_not_horizontal`**, **`pcb_off_rail_xy`**, **`pcb_fallen_below_rail`**, **`pcb_dropped`**, and **`arm_idle`** (`_ARM_IDLE_MIN_STEPS`, `_ARM_IDLE_MAX_ABS_VEL_RAD_S`) if episodes reset too aggressively—or not enough when the PCB slips / the policy freezes.

### Adding forward-push rewards

`RewardsCfg` includes approach to the **push-face / short-edge** center, **`grasp_short_edge`**, **forward-only** slide, **`no_central_top_bottom_face`** (penalty for tool midpoint near top/bottom over the inner ~60 %×60 % of the face, leaving a 20 % edge band per side), and regularizers. Observations include **`gripper_opening`**, **`ee_thickness_offset`**, and **`pinch_orientation_cos`** (|cos| for opening∥thickness and finger-line∥short edge). Rewards add **`pinch_thickness_align`**, **`pinch_orientation_flat_edge`** (gated near 단변), and **`no_open_side_rub`**. If the fingertip geometry in your USD differs, tune ``gripper_left``/``right`` or the cross-product convention in ``mdp_custom.gripper_pinch_orientation_flat_edge_reward``.

### If mean reward plateaus (policy idles near the edge)

1. In `RewardsCfg`, balance **`approach_trailing_edge`** vs **`push_velocity`** vs **`action_rate_penalty`** (defaults are tuned so slide reward can compete with a small residual distance).
2. In PPO (`agents/rl_games_ppo_cfg.py`), raise **entropy** slightly or decay it more slowly if the policy collapses early.
3. Log **per-term rewards** in TensorBoard if available, to see which term is flat.

---

## Troubleshooting

- **Body name errors:** Match `SceneEntityCfg("robot", body_names=...)` to link names in your loaded robot USD (e.g. `link_6`, `gripper_left`, `gripper_right`).
- **Penetration at reset:** Slightly increase vertical `pos_offset`, soften edge offsets, or check gripper initial opening.
---

## Quick workflow

1. Match `_MAG_*` / `_PCB_INIT_*` to your combined USD.
2. Add rail-forward rewards and success criteria as needed.
3. Train and monitor TensorBoard.
4. Tune reset and termination thresholds if needed.
