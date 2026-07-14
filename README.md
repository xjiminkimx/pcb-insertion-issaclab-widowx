# WidowX PCB Approach + Slide — Isaac Lab

Manager-based RL task for a **WidowX** arm that **open-jaw straddles** the PCB trailing short edge, then **pushes +Y** along the conveyor/guide rails toward the magazine slot.

Training uses **rl-games** (PPO). Checkpoints and TensorBoard logs are written under this package’s `logs/` when you use the workspace scripts.

| Robot | Task ID | Description |
|--------|---------|-------------|
| WidowX (`usd_model/usd_robot/wxai/wxai_follower.usd`) | **`Isaac-WidowX-PCB-Push-v0`** | Open-jaw approach + +Y push / slide to magazine |
| | `Isaac-WidowX-PCB-Straddle-v0` | Deprecated alias of Approach |

**Pipeline (single phase):**

1. Reset PCB on guide rails (domain-randomized XY / yaw).
2. Gripper held open at **40 mm** span (`hold_gripper_open`).
3. Policy approaches trailing edge (`finger_proximity` / mid-thickness / gap shaping).
4. After push gate (closedness), credit +Y PCB motion (`push_axis_velocity`, progress, milestones).
5. Episode ends on **`slide_success`** (leading edge near magazine terminus) or safety terminations.

Legacy Grasp → Slide → Insert chaining scripts/configs may still exist in the tree, but the **active registered envs are Approach + Slide**.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers `Isaac-WidowX-PCB-Approach-v0` (+ deprecated Straddle alias). |
| `widowx_pcb_env_cfg.py` | Scene, Approach rewards / events / terminations, geometry constants. |
| `mdp_custom.py` | Geometry, push-gated rewards, debug curriculum, reset helpers. |
| `usd_model/env_v6/` | `assembly_6.urdf` + `convert_to_usd.py` → **`pcb_insertion_env.usd`** (runtime fixture). |
| `usd_model/usd_robot/wxai/` | WidowX follower USD. |
| `scripts/train_approach.sh` / `play_approach.sh` | Primary train / play entry points. |
| `scripts/train_straddle.sh` / `play_straddle.sh` | Wrappers → push scripts. |
| `scripts/collect_approach_states.py` | Optional terminal-state collection for downstream chaining. |
| `agents/` | `WidowXPcbApproachPPOCfg`, TensorBoard helper, workspace paths. |
| `data/` | Terminal-state `.npz` buffers (see `data/README.md`). |
| `logs/rl_games/widowx_pcb_approach/` | Training summaries + `nn/*.pth` checkpoints. |

---

## Prerequisites

- [Isaac Sim](https://developer.nvidia.com/isaac-sim) and **Isaac Lab** installed (same layout as the parent `IsaacLab` repo).
- Conda (or equivalent) env that can run Isaac Lab RL scripts (example: `isaac-sim`).

---

## Train (rl-games PPO)

**Recommended:** run from this package so logs land under `logs/rl_games/widowx_pcb_approach/`.

```bash
conda activate isaac-sim   # or your env name
cd /path/to/widowx_pcb

# Triangle-mesh fixture colliders need larger GPU collision buffers.
# Prefer 2048 envs if you see PhysX collisionStackSize overflow at 4096.
bash scripts/train_approach.sh --num_envs 2048 --headless
```

Deprecated alias:

```bash
bash scripts/train_straddle.sh --num_envs 2048 --headless
```

If you previously trained under Isaac Lab root:

```bash
bash scripts/sync_logs_from_isaaclab.sh
```

Alternative (logs under `<IsaacLab>/logs/`):

```bash
cd /path/to/IsaacLab
python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-WidowX-PCB-Push-v0 --num_envs 2048 --headless
```

PPO knobs: `agents/rl_games_ppo_cfg.py` → `WidowXPcbApproachPPOCfg`  
(`name: widowx_pcb_approach`, `max_epochs: 150`, `entropy_coef: 1e-2`, …).

---

## Evaluate / play

Checkpoints: `logs/rl_games/widowx_pcb_approach/nn/`  
Default best model: `widowx_pcb_approach.pth`

**Important:** Isaac Lab `play.py` resolves `logs/rl_games/...` relative to **cwd**. Run from this package root.

```bash
cd /path/to/widowx_pcb
bash scripts/play_approach.sh --num_envs 1
```

Specific checkpoint:

```bash
bash scripts/play_approach.sh \
  --num_envs 1 \
  --checkpoint logs/rl_games/widowx_pcb_approach/nn/last_widowx_pcb_approach_ep_30_*.pth
```

Without `--checkpoint`, play loads the best `nn/widowx_pcb_approach.pth`.  
`--use_last_checkpoint` picks the latest epoch file.  
Add `--real-time` for wall-clock playback.

---

## Task design (Approach)

### Actions

- Arm: effort on `joint_[0-5]` (`ActionsCfgApproach`).
- Gripper: **not** policy-controlled; PD-held open every step (`hold_gripper_open` at 40 mm span).

### Rewards (`RewardsApproachCfg`)

| Term | Role (approx.) |
|------|----------------|
| `finger_proximity` | Dense approach to ±20 mm trailing-edge finger targets (σ ≈ 35 mm) |
| `tip_mid_thickness` | Pull pads to PCB mid-thickness (edge height) |
| `lateral_gap` | Symmetric jaw–PCB width gaps |
| `pcb_tilt_penalty` | Discourage knock-over |
| `leading_edge_push_progress` | Gated +Y leading-edge progress |
| `push_axis_velocity` | Gated +Y PCB velocity |
| `slide_travel_milestone` | One-shot sparse bonuses at travel fractions |
| `goal_lead_proximity` | Soft proximity to goal leading-edge pose |
| `pcb_yaw_alignment` | Keep PCB yaw aligned with push axis |
| `slide_success_bonus` | Large sparse bonus on success |
| `action_rate_penalty` | Smooth actions |

Push credit is gated on **finger-target closedness** (same family as straddle geometry), not the old `pcb_between_gripper_fingers` quality score.

### Terminations (`TerminationsApproachCfg`)

- `slide_success` — leading edge in success box (see `_slide_success_params`)
- `pcb_tilt_excessive`, `pcb_long_axis_not_horizontal`
- `pcb_fallen_below_rail`, `pcb_moving_backward`
- `time_out` (`episode_length_s = 8.0 s`)

### Debug / TensorBoard curriculum

`Curriculum/approach_gripper_debug/*` (via `approach_gripper_debug_curriculum`):

- `closedness` / `closedness_tight` — episode-end batch means (proximity vs tight σ)
- `closedness_peak`, `straddle_success_frac`
- `push_gate_open_frac`, `lead_vy`, distances in mm

---

## Fixture USD (`env_v6`)

Runtime asset loaded by the scene:

```text
usd_model/env_v6/pcb_insertion_env.usd
```

Regenerate after changing the converter:

```bash
cd usd_model/env_v6 && python3 convert_to_usd.py
```

### What is included / excluded

| Included | Notes |
|----------|--------|
| Magazine | Triangle-mesh collider (`physics:approximation = none`) |
| Guide rails (`Part_1_7`) | Horizontal PCB support — triangle mesh |
| Side belts (`Part_1_2`) | Triangle mesh |
| Stand / frame | `convexDecomposition` |

| Excluded | Reason |
|----------|--------|
| Chip / PCB mesh | Spawned as a separate cuboid in the env |
| Short axles (`Part_1_3`) | Decorative |
| Raised side rail-guides (`Part_1_4`, `Part_1_6`) | Tall posts beside conveyor — removed so they do not block approach |

Contact / rest offsets on fixture meshes default to **0.0001 m**.  
PCB cuboid uses the same order of magnitude in `widowx_pcb_env_cfg.py`.

### Support height vs “floating” look

- Side **belt** top is lower than guide **rail** tops in the `env_v6` assembly — the PCB rests on the **rails**, not the belt surface.
- A few mm gap above the belt in the viewport is normal when the board is seated on the rails (not only `rest_offset`).
- Spawn height is driven by `_CONVEYOR_SURFACE_Z` → `_PCB_CENTER_Z_ENV` in `widowx_pcb_env_cfg.py`. Align that with **rail support**, not the belt visual alone.

### PhysX GPU buffers

Full triangle meshes × many envs can overflow `gpu_collision_stack_size` (contacts dropped → tunneling / floaty PCB). Push cfg sets:

```text
gpu_collision_stack_size = 2**29   # ~512 MB
```

Prefer **`--num_envs 2048`** (or lower) if overflow returns at 4096.

**Do not** casually switch magazine to SDF without retuning `sdf_margin` / resolution: a narrow slot can behave “thicker” and eject the PCB.

---

## TensorBoard

```bash
cd /path/to/widowx_pcb
python agents/monitor_tensorboard.py --port 6006
```

Or:

```bash
tensorboard --logdir=logs --port=6006
```

Open <http://127.0.0.1:6006>. Primary run: `rl_games/widowx_pcb_approach/summaries`.

### Useful scalars

| Tag / area | Meaning |
|------------|---------|
| Episodic reward | Overall learning progress |
| `Episode_Reward/<term>` | Per-term Approach rewards |
| `Episode_Termination/slide_success` | Success rate |
| `Curriculum/approach_gripper_debug/closedness` | Approach quality (proximity σ) |
| `Curriculum/approach_gripper_debug/closedness_tight` | Same gate family as push unlock (tight σ) |
| `Curriculum/approach_gripper_debug/push_gate_open_frac` | Fraction of steps with push gate open |
| `Curriculum/approach_gripper_debug/lead_vy` | PCB +Y velocity snapshot |

Sparse terms (e.g. `slide_travel_milestone`) average over envs and are scaled by `dt` / episode length in Isaac Lab logging — they can look near-zero even when firing on a few envs.

---

## Geometry / env knobs (`widowx_pcb_env_cfg.py`)

| Constant | Role |
|----------|------|
| `_MAG_POS` / `_MAG_ROT_WXYZ` | Fixture placement (−90° Z maps root −X → world +Y push) |
| `PUSH_AXIS_WORLD` | `(0, 1, 0)` |
| `_CONVEYOR_SURFACE_Z` | Spawn / height reference (align with rail top in practice) |
| `_PCB_INIT_POS` | PCB reset pose (lane X, Y before slot, Z on support) |
| `_APPROACH_OPEN_WIDTH_M` | Open span target (40 mm jaw span / carriage scale) |
| `_APPROACH_FINGER_OFFSET_M` | ±20 mm trailing-edge finger targets |
| `_APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD` | Closedness level used with push gating |
| `_SLIDE_SUCCESS_LEAD_Y_ENV` | Leading-edge success / milestone terminus along +Y |
| `_SLIDE_TRAVEL_MILESTONE_FRACTIONS` | One-shot travel tiers |

Align magazine slot constants (`_MAG_Y_NEAR_FACE_ENV`, success XY) with Isaac Sim measurements after moving the fixture.

---

## Optional: collect push terminal states

For Sequential Dexterity-style downstream phases (if enabled again later):

```bash
bash scripts/collect_push_states.sh \
  --checkpoint logs/rl_games/widowx_pcb_approach/nn/widowx_pcb_approach.pth \
  --num_envs 256 --num_states 2000 --headless
```

See `data/README.md` for `.npz` field layout.

---

## Branching model

- **`main`** — release-oriented working line.
- **`dev`** — day-to-day integration; merge into `main` when ready.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|-------------------|
| PhysX `collisionStackSize` overflow | Raise `gpu_collision_stack_size` further, or lower `--num_envs` (2048 recommended with triangle magazine). |
| PCB looks above belt but on rails | Expected if seated on rails. Check rail contact / `_CONVEYOR_SURFACE_Z`, not belt visual. |
| PCB sinks / tunnels | Contact drop from GPU overflow, or spawn Z too low vs rail collision. |
| Slide never starts (low `push_gate_open_frac`) | Closedness under gate threshold — watch `closedness_tight` / distances in TB. |
| Idles at trailing edge | Approach rewards dominate push terms — rebalance weights / idle penalty / gate. |
| Checkpoint not found in play | Run play from `widowx_pcb` cwd, not Isaac Lab root. |

---

## License

See `LICENSE` in this package.
