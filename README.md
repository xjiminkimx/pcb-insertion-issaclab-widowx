# WidowX PCB Approach + Insert — Isaac Lab

Manager-based RL task for a **WidowX** arm standing **beside** a conveyor. The task is split into two
independently trained phases, chained through a terminal-state buffer:

1. **Approach** — open-jaw **straddle** of the PCB trailing short edge (no board motion allowed).
2. **Insert** — push the straddled board **+Y** along the guide rails into the magazine slot.

Both phases use **task-space variable impedance control** (Isaac Lab OSC) and are trained with
**rl-games** PPO. Checkpoints and TensorBoard logs land under this package's `logs/`.

| Task ID | Env cfg | Phase |
|---------|---------|-------|
| **`Isaac-WidowX-PCB-Approach-v0`** | `WidowXPcbApproachEnvCfg` | 1 — straddle |
| **`Isaac-WidowX-PCB-Insert-v0`** | `WidowXPcbInsertEnvCfg` | 2 — push into slot |

Robot USD: `usd_model/usd_robot/wxai/wxai_follower.usd`.

---

## Pipeline

```text
train_approach.sh  ->  collect_approach_states.sh  ->  train_insert.sh
   (phase 1)            data/approach_terminal_states.npz     (phase 2)
```

**Phase 1 — Approach** (`episode_length_s = 3.0`)

1. PCB spawns on the rails with randomized X / yaw; gripper is PD-held open at a **30 mm** span.
2. Arm resets to a fixed FK-solved home posture beside the belt.
3. Policy drives the pads onto the trailing short edge: pads at mid-thickness, jaws straddling the
   board across its **width**, wrist trailing the pads along +Y.
4. Terminates on `approach_success` (closedness **and** tip-mid-thickness above threshold) — any +Y
   board displacement is penalised, sliding is *not* part of this phase.

**Phase 2 — Insert** (`episode_length_s = 8.0`)

1. Reset replays a random Approach terminal state (arm joints + PCB placed relative to the gripper).
2. Policy pushes the trailing edge +Y with the jaws held open, keeping the board flat and on-lane.
3. Terminates on `insert_success` (leading edge seated near the magazine back wall) or on failure.

> The terminal-state buffer is tied to the base placement and the Approach policy that produced it.
> After moving the robot base or retraining Approach, re-run `collect_approach_states.sh` before
> training Insert, or Insert will reset into a pose that no longer exists.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers `Isaac-WidowX-PCB-Approach-v0` and `Isaac-WidowX-PCB-Insert-v0`. |
| `widowx_pcb_env_cfg.py` | Scene, geometry constants, per-phase actions / observations / rewards / events / terminations. |
| `mdp_custom.py` | Task-space impedance action term, geometry helpers, reward & termination functions, reset helpers. |
| `usd_model/env_v7/` | `assembly_1.urdf` + `convert_to_usd.py` → **`pcb_insertion_env.usd`** (runtime fixture). |
| `usd_model/usd_robot/wxai/` | WidowX follower USD. |
| `scripts/train_approach.sh` / `play_approach.sh` | Phase 1 train / play. |
| `scripts/train_insert.sh` / `play_insert.sh` | Phase 2 train / play. |
| `scripts/play_chain.sh` / `play_chain.py` | Live Approach→Insert chain play (optional success gates). |
| `scripts/eval_success.sh` / `eval_success.py` | Success-rate eval for Approach / Insert / Chain. |
| `scripts/collect_approach_states.py` / `.sh` | Roll out an Approach checkpoint and store terminal states. |
| `scripts/diag_side_base_home_pose.py` | FK search / inspection of the reset posture and base placement. |
| `scripts/diag_ee_box.py` | Instruments the EE translation / rotation boxes; measures droop, wrist pitch, clearance. |
| `agents/` | PPO configs, rl-games patches, TensorBoard helper, workspace paths. |
| `data/` | Terminal-state `.npz` buffer (see `data/README.md`). |
| `logs/rl_games/widowx_pcb_{approach,insert}/` | Summaries + `nn/*.pth` checkpoints. |

---

## Prerequisites

- [Isaac Sim](https://developer.nvidia.com/isaac-sim) and **Isaac Lab** installed (same layout as the
  parent `IsaacLab` repo).
- Conda (or equivalent) env that can run Isaac Lab RL scripts (example: `isaac-sim`).

---

## Train

Run from this package so logs land under `logs/rl_games/…`.

```bash
conda activate isaac-sim
cd /path/to/widowx_pcb

# Phase 1
bash scripts/train_approach.sh --num_envs 2048 --headless

# Chain: roll out the Approach checkpoint into a terminal-state buffer
bash scripts/collect_approach_states.sh \
  --checkpoint logs/rl_games/widowx_pcb_approach/nn/widowx_pcb_approach.pth \
  --num_envs 256 --num_states 2000 --headless

# Phase 2
bash scripts/train_insert.sh --num_envs 2048 --headless
```

Both train scripts accept `--clean-logs` (drop TensorBoard summaries, keep checkpoints) and
`--clean-all` (drop the whole phase log dir).

PPO knobs live in `agents/rl_games_ppo_cfg.py`:

| Config | `name` | `max_epochs` | `horizon_length` | `clip_actions` |
|--------|--------|--------------|------------------|----------------|
| `WidowXPcbApproachPPOCfg` | `widowx_pcb_approach` | 150 | 128 | 0.25 |
| `WidowXPcbInsertPPOCfg` | `widowx_pcb_insert` | 120 | 256 | 0.25 |

`agents/rl_games_logstd_safety.py` is monkey-patched in at import time to stop `Normal.sample` from
failing when `exp(log_std)` underflows.

---

## Evaluate / play

**Important:** Isaac Lab `play.py` and the chain script resolve `logs/rl_games/...` relative to
**cwd** — run from this package root.

### Success-rate evaluation (`eval_success`)

```bash
cd /path/to/widowx_pcb

# Parallel Approach / Insert (deterministic actions)
bash scripts/eval_success.sh approach --num_episodes 200 --num_envs 256 --headless
bash scripts/eval_success.sh insert   --num_episodes 200 --num_envs 256 --headless

# Live Approach→Insert chain (sequential, num_envs=1)
bash scripts/eval_success.sh chain --num_episodes 50 --headless

# All three, then JSON under logs/rl_games/eval/
bash scripts/eval_success.sh all --num_episodes 100 --num_envs 256 --headless
```

| Flag | Default | Meaning |
|------|---------|---------|
| `mode` | required | `approach` / `insert` / `chain` (shell also accepts `all`) |
| `--num_episodes` | `100` | Completed episodes (or full chains) to score |
| `--num_envs` | `64` | Parallel envs for approach/insert (`chain` forces 1) |
| `--checkpoint` | `weight_saved/` then `nn/` | Single-phase checkpoint |
| `--approach_checkpoint` / `--insert_checkpoint` | same defaults | Chain (or phase) checkpoints |
| `--out` | `logs/rl_games/eval/<mode>_*.json` | JSON summary path |
| `--seed` | `-1` | Fresh seed each run (printed for reproducibility) |

```bash
cd /path/to/widowx_pcb
bash scripts/play_approach.sh --num_envs 1
bash scripts/play_insert.sh --num_envs 16
```

Without `--checkpoint`, `play_approach` / `play_insert` load the best `nn/<name>.pth`;
`--use_last_checkpoint` picks the latest epoch file; `--real-time` gives wall-clock playback.

### Live chain play (`play_chain`)

`scripts/play_chain.sh` runs **Approach → Insert in one session** with a live straddle handover
(no `collect_approach_states` step). On Approach success it freezes that pose into a one-row
buffer, reconfigures the same SimulationContext to Insert managers, and continues with the Insert
policy. Closing one gym env and `gym.make`-ing the other in the same Kit process hangs, so the
script keeps a single env and only swaps MDP managers.

**Example** (GUI + Insert debug prints + half-speed video + looser Approach gates):

```bash
bash scripts/play_chain.sh --gui --debug \
  --approach_checkpoint logs/rl_games/widowx_pcb_approach/weight_saved/widowx_pcb_approach.pth \
  --insert_checkpoint logs/rl_games/widowx_pcb_insert/weight_saved/widowx_pcb_insert_05mm_dent.pth \
  --video --closedness 0.3 --min_tip_down_deg 14
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--approach_checkpoint` | `weight_saved/` then `nn/` `widowx_pcb_approach.pth` | Approach policy |
| `--insert_checkpoint` | `weight_saved/` then `nn/` `widowx_pcb_insert.pth` | Insert policy |
| `--gui` | off (headless) | Show Isaac Sim viewport |
| `--video` | off | Record full chain as one **0.5×** mp4 (`num_envs=1`, cameras on) |
| `--video_dir` | `logs/rl_games/widowx_pcb_chain/videos/play` | Output directory for mp4 |
| `--closedness Q` | training (`0.40`) | Approach success closedness ≥ Q |
| `--min_tip_down_deg DEG` | training (`17.5`) | Tip-down pitch ≤ −DEG |
| `--tip_mid Q` | training (`0.60`) | Tip mid-thickness index ≥ Q |
| `--no_pitch_gate` / `--no_tip_mid_gate` | off | Drop that Approach success conjunct |
| `--debug` | off | Print Insert leading-edge vs success-box each `--print_every` steps |
| `--episodes N` | `1` | How many full Approach→Insert chains |

Video notes:

- Playback fps = `1 / (2 · step_dt)` ≈ **62.5** → wall-clock is half of sim time.
- Camera is the elevated corner view (`eye=(0.87, 0.88, 0.60)`, `lookat=(0.25, 0.35, 0.26)`,
  `1920×1080`) — diagonal baseplate with robot left / magazine right.
- Output filename: `chain_ep<N>_<timestamp>_halfspeed.mp4`.

Other useful invocations:

```bash
# Training success gates, GUI only:
bash scripts/play_chain.sh --gui

# Headless half-speed recording with defaults:
bash scripts/play_chain.sh --video --headless

# Drop pitch gate for demos that never quite tip down enough:
bash scripts/play_chain.sh --gui --no_pitch_gate --debug
```

---

## Scene and geometry

| Item | Value |
|------|-------|
| PCB | 240 × 78.5 × 1 mm cuboid, spawned separately from the fixture |
| Push axis | `PUSH_AXIS_WORLD = (0, 1, 0)` — world +Y, toward the magazine |
| Magazine slot | entry face at env Y ≈ 0.207, back wall at ≈ 0.472 |
| Robot base | `(-0.25, -0.25, 0.0025)`, **identity rotation** |
| Sim | `dt = 0.002`, `decimation = 4` → 125 Hz control (`step_dt = 0.008 s`) |

### Why the base sits beside the belt

The base is **not** yawed and **not** behind the push line. With the old behind-the-belt placement the
whole slide was a radial extension of the arm along +Y: the shoulder/elbow had to keep stretching as
the board advanced, walking the arm toward its reach limit exactly when insertion needed the most
force. Beside the belt, the push is executed mostly as a **joint_0 (base yaw) sweep** at a steady
mid-range configuration.

Keeping the base rotation at identity is what makes **base-local task axes equal world axes**, so the
OSC per-axis tables below mean what they say:

```text
tx = world +X = lateral (across the belt)      rx = pitch about world X (wrist tip up/down)
ty = world +Y = push / insertion axis          ry = roll about the push axis (pad levelness)
tz = world +Z = vertical                       rz = yaw about vertical (board skew)
```

The reset posture (`_ROBOT_HOME_JOINT_POS`) is FK-solved, not hand-tuned, by
`scripts/diag_side_base_home_pose.py`: with the base beside the belt the shoulder plane points across
the belt, but the gripper must still present the straddle geometry the Approach rewards score (jaws
opening along the PCB width = world X, both pads level, wrist trailing the pads along +Y). The wrist
therefore absorbs ~90° of yaw, and since the Approach orientation box caps cumulative commanded
rotation, **whatever orientation error the reset pose has is permanent** — the policy cannot twist
the wrist into place during an episode.

---

## Action space (both phases)

18 dimensions per step, consumed by `WidowXTaskSpaceImpedanceAction` (`mdp_custom.py`):

| Slice | Meaning |
|-------|---------|
| `[0:6]` | `pose_rel` — Δposition (×`position_scale`) and Δrotation (×`orientation_scale`) |
| `[6:12]` | Per-axis task stiffness `K`, remapped from `[-1,1]` into the per-axis limits |
| `[12:18]` | Per-axis damping ratio `ζ` |

The gripper is **not** policy-controlled in either phase; it is PD-held open every step.

### The two "boxes" (read this before touching the action config)

Isaac Lab's OSC recomputes `pose_rel` targets as **`current pose ⊕ delta` every control step**, so
there is no absolute anchor. Two consequences bit this task hard, and both are now handled by
absolute-from-reset clamps anchored by the `store_reset_ee_pose` / `store_insert_reset_ee_pose` reset
events:

- **`task_position_box_enabled`** — without it, gravity-compensation residual sinks the arm and the
  sunk pose becomes the new setpoint (the "쳐짐" ratchet). Measured with zero actions: pad Z −6 mm at
  0.2 s, −38 mm at 2 s, −71 mm at 3.8 s. With the box the offset stays within a few mm.
- **`task_orientation_box_enabled`** — the same ratchet in rotation. Measured on Insert with the
  position box already on: wrist pitch ran −13° → −25° in 1.6 s and kept falling linearly, i.e. the
  gripper head simply flops down over an episode. **Stiffness cannot fix either one** — the tracking
  error is ~0 by construction, so the impedance spring never generates a restoring term.

| Parameter | Approach | Insert |
|-----------|----------|-------|
| `position_scale` / `orientation_scale` | 0.15 / 0.15 | 0.04 / 0.10 |
| lateral (X) half-range | ±0.08 m | ±0.01 m |
| vertical (Z) half-range | ±0.06 m | ±0.01 m |
| push (Y) offset range | −0.05 … +0.20 m | 0 … +0.60 m |
| rotation bound | ±0.15 rad (≈ ±8.6°), all three axes | per-axis (see below) |

Note that `_ARM_TASK_ORIENTATION_MAX_DEV_RAD = 0.15` is **radians**, i.e. ≈ 8.6°, not the 15° the
inline comment claims. That bound is tight enough that Approach cannot correct the ~8° jaw roll it
inherits from the reset posture, which is why Insert starts with the pads a few mm out of level.

Insert needs an **asymmetric per-axis** rotation bound, because the wrist must be free to pitch tens of
degrees down for rail clearance while roll and yaw stay pinned near the straddle orientation. A
single symmetric scalar cannot express that, hence `orientation_dev_limits_per_axis`:

```python
_INSERT_ORIENTATION_DEV_LIMITS_PER_AXIS = (
    (-0.30, 0.09),   # rx  pitch: negative = tip-DOWN
    (-0.10, 0.10),   # ry  roll about the push axis
    (-0.12, 0.12),   # rz  yaw about vertical
)
```

Measured effect on Insert: zero-action equilibrium settles at **−18°** (reset −13° plus ~5° of static
gravity sag) and stays there; full tip-down command saturates at **−35°**; full tip-up at −13°.

### Selective compliance (Insert)

`_ARM_TASK_INSERT_STIFFNESS_LIMITS_PER_AXIS` caps `K` per axis regardless of what the policy asks for:
`ty` (push) is stiffest at 250–2500 so the board actually breaks static friction, `tx`/`rz` are softer
so it can self-align in the slot, and `tz`/`rx`/`ry` are soft but **never free** — a fully
uncontrolled axis has literally zero restoring torque and once `ty` could deliver real force, the
coupling into those axes diverged and crashed PhysX.

---

## Phase 1 — Approach

### Rewards (`RewardsApproachCfg`)

All weights are multiplied by `step_dt = 0.008 s` before entering the return. One-shot
`approach_success_bonus` at 20000 → **+160** effective.

| Term | Weight | MDP function | Index / output |
|------|--------|--------------|----------------|
| `trailing_face_approach` | 150 | `straddle_trailing_face_bounded_approach_reward_fade_near_success` | `[0, 1]` per jaw, min |
| `tip_mid_thickness` | 200 | `straddle_tip_mid_thickness_shaping_gated_fade_near_success` | `[0, 1]` |
| `finger_proximity` | 150 | `straddle_finger_trailing_width_proximity_fade_near_success` | `[0, 1]` per jaw |
| `lateral_gap` | 120 | `straddle_lateral_gap_shaping_fade_near_success` | `[0, 1]` |
| `between_fingers` | 60 | `pcb_between_gripper_fingers_fade_near_success` | `[0, 1]` |
| `wrist_tip_down` | 30 | `gripper_wrist_carriage_tip_down_pitch_shaping_gated` | `[0, 1]` ramp |
| `pcb_forward_push_penalty` | −100 | `pcb_forward_push_displacement_indicator` | `{0, 1}` |
| `approach_success_bonus` | 20000 | `approach_success_bonus_reward` | one-shot |
| `action_rate_penalty` | −0.002 | `action_rate_l2` | L2 |
| `wrist_pitch_deg_debug` | 1e-10 | `gripper_wrist_pitch_deg_signed_obs` | deg |
| `approach_gripper_debug_monitor` | 1e-10 | `approach_gripper_debug_monitor_reward` | debug |

Dense straddle terms use `*_fade_near_success`: full credit until `closedness_tight` reaches
`_APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD`, then linear fade to `min_scale = 0.05` by +0.15.
Terminal states feed Phase 2, so success also gates tip-mid and tip-down pitch.

#### Reward details and parameters

| Term | Role | Key parameters (defaults in `widowx_pcb_env_cfg.py`) |
|------|------|------------------------------------------------------|
| **`trailing_face_approach`** | Bell along +Y toward the trailing face; **only term that decays on overshoot** past the edge. | `approach_std_m = 0.050`, `overshoot_std_m = 0.008`, `target_along_m = 0`, finger targets ±15 mm (`_APPROACH_GAP_*_M`), fade at closedness 0.40→0.55 |
| **`tip_mid_thickness`** | Pull pad tips to the mid-thickness plane (not top/bottom face). | `std = 0.012 m`, gated until `min_closedness = 0.3` (proximity σ = 0.10 m) |
| **`finger_proximity`** | Far-field magnet toward ±15 mm trailing-edge targets. | `std = 0.10 m` (wide); success/debug uses `0.035 m` |
| **`lateral_gap`** | Explicit L/R width-gap shaping — fixes jaw off-centre on the PCB. | targets ±15 mm, `width_gap_sigma_m = 0.035 m` |
| **`between_fingers`** | PCB between jaws × near trailing edge × width gaps (`between_fingers_q`). | `proximity_sigma_m = 0.10`, `width_sigma_m = 0.025`; along factor is one-sided past the face |
| **`wrist_tip_down`** | Signed tip-down only (jaw below wrist). | target **25°**, max **35°**; closedness gate **0.30→0.45**; tip-mid gate **0.20→0.60** |
| **`pcb_forward_push_penalty`** | Forbid +Y board motion during approach. | `max_displacement_m = 0.005` from spawn Y |
| **`approach_success_bonus`** | Sparse win — same gates as `approach_success`. | see success table below |

### Success and terminations

`approach_success` requires **all** of:

| Gate | Threshold | Parameter |
|------|-----------|-----------|
| `closedness_tight` | ≥ **0.40** | `_APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD`, σ = **35 mm** |
| `tip_mid_thickness` | ≥ **0.60** | `_APPROACH_SUCCESS_TIP_MID_THICKNESS_THRESHOLD`, σ = **12 mm** |
| tip-down pitch | ≥ **17.5°** | `_APPROACH_SUCCESS_MIN_TIP_DOWN_DEG` |

Failures: `pcb_tilt_excessive`, `pcb_long_axis_not_horizontal` (`|long_axis · Z| > 0.50`),
`pcb_fallen_below_rail`, `pcb_moving_backward`, `time_out`.

### Debug scalars

`Curriculum/approach_gripper_debug/*` — use `closedness_tight` (success σ), not loose
`closedness` (10 cm σ). Also: `pitch_deg_*`, per-side gaps (mm), `straddle_success_frac`.

---

## Phase 2 — Insert

### Rewards (`RewardsInsertCfg`)

All weights × `step_dt = 0.008 s`. One-shot `insert_success_bonus` at 25000 → **+200**.
Phase is sized around a **push-vs-freeze ledger**: static income must stay below
`alive_penalty` (−42/s).

| Term | Weight | MDP function | Index / output |
|------|--------|--------------|----------------|
| `alive_penalty` | −42 | `mdp.is_alive` | per step |
| `failure_penalty` | −10000 | `mdp.is_terminated_term` | one-shot (−80 eff.) |
| `tip_mid_thickness` | 12 | `straddle_tip_mid_thickness_shaping` | `[0, 1]` |
| `tip_under_penalty` | −20 | `gripper_tip_under_pcb_penalty` | `[0, 1]` |
| `pcb_edge_parallel_penalty` | −20 | `pcb_edge_axis_parallel_penalty` | `[0, 2]` sum |
| `leading_edge_push_progress` | 50 | `pcb_leading_edge_push_axis_approach_progress_seated` | `[0, 1]` Δy |
| `push_axis_velocity` | 50 | `pcb_push_axis_velocity_reward_seated` | `[0, 1]` vy |
| `insert_travel_milestone` | 2000 | `insert_leading_edge_travel_milestone_bonus` | tier count |
| `jaw_rail_clearance` | 25 | `gripper_jaw_rail_clearance_shaping` | `[0, 1]` |
| `insert_success_bonus` | 25000 | `insert_success_bonus_reward` | one-shot |
| `wrist_pitch_deg_debug` | 1e-10 | `gripper_wrist_pitch_deg_signed_obs` | deg |
| `jaw_clearance_mm_debug` | 1e-10 | `jaw_rail_clearance_mm_obs` | mm |
| `board_travel_mm_debug` | 1e-10 | `insert_leading_edge_travel_mm_obs` | mm |
| `board_lane_drift_mm_debug` | 1e-10 | `insert_leading_edge_lane_drift_mm_obs` | mm |

**Disabled:** `goal_lead_proximity`, `pcb_yaw_alignment`, `wrist_tip_down` (clearance scored
directly via `jaw_rail_clearance`).

#### Reward details and parameters

| Term | Role | Key parameters |
|------|------|----------------|
| **`leading_edge_push_progress`** | Per-step +Y travel of leading-edge centre: `clamp(Δy,0,max)/max`. | `max_step_m = 0.0006` (~0.075 m/s at saturation); gated on tip-mid **≥ 0.55** and jaw clearance ramp **14→28 mm** |
| **`push_axis_velocity`** | +Y root velocity normalised to ref speed. | `ref_speed_m_s = 0.062`, `min_push_speed_m_s = 0.005`; same seat/clearance gate as progress |
| **`insert_travel_milestone`** | One-shot per travel fraction (latched). | fractions **12 / 25 / 50 / 75 / 87.5 / 95 %** to `_INSERT_SUCCESS_LEAD_Y_ENV`; lane gate X/Z drift **≤ 20 mm** |
| **`tip_mid_thickness`** | Seating attractor on trailing-edge mid-thickness (+2 mm Z offset). | `std = 8 mm`, `thickness_target_offset_m = 2 mm` |
| **`tip_under_penalty`** | Symmetric tip off the thickness band (under or over). | half-thickness 0.5 mm, cap **20 mm** excess |
| **`pcb_edge_parallel_penalty`** | Board square + flat in lane (corner geometry). | in-plane dead **2 mm**, cap **20 mm**; lift dead **2 mm**, cap **10 mm** |
| **`jaw_rail_clearance`** | Lower-jaw height above rail plane × tip-mid seating × travel gate. | target **28 mm**; travel gate **0→30 mm** (floor 0.45); seating σ = 8 mm |
| **`alive_penalty`** | Per-step cost while alive — keeps freeze net-negative. | weight **−42** vs ~37/s static ceiling (tip_mid + clearance) |
| **`failure_penalty`** | One-shot on `pcb_fallen_below_rail` or `pcb_long_axis_not_horizontal`. | effective **−80**; Insert long-axis limit **0.25** |
| **`insert_success_bonus`** | Sparse win when `insert_success` fires. | leading-edge XY box at magazine back; gripper gap check optional |

Rough ledger:

| Behavior | Approx. rate |
|----------|--------------|
| Freeze, well seated | ~37/s income − 42/s alive ≈ **−5/s** |
| Seated push at pace | ~100/s from progress + velocity |
| All milestones (6 tiers) | ~**19/s** amortized (`2000` weight, `dt`-scaled) |
| Success / hard failure | **+200** / **−80** once |

### Success and terminations

`insert_success`: leading-edge centre in XY box near magazine back wall
(`_INSERT_SUCCESS_TARGET_LEAD_XY_ENV`, tolerance **10 mm × 20 mm**). Failures:
`pcb_fallen_below_rail`, `pcb_long_axis_not_horizontal` (`|long_axis · Z| > 0.25`), `time_out`.

Debug mm terms: divide `Episode_Reward/<term>` by `1e-10` to read raw geometry.

---

## Diagnostics

Both diagnostic scripts are FK / instrumentation tools, not training entry points.

```bash
# Reset-posture and base-placement geometry (jaw axis vs PCB width, pad levelness, reach, joint margins)
python -u scripts/diag_side_base_home_pose.py --inspect --num_envs 4 --headless
python -u scripts/diag_side_base_home_pose.py --num_envs 4096 --headless      # search a new home pose

# EE boxes: is the clamp running, what is it anchored to, how much does the arm droop / flop?
python -u scripts/diag_ee_box.py --num_envs 4 --steps 200 --headless           # Approach
python -u scripts/diag_ee_box.py --insert --num_envs 4 --steps 400 --headless   # Insert + wrist pitch + clearance
python -u scripts/diag_ee_box.py --insert --num_envs 1 --steps 5000 --rot_cmd -1 0 0   # GUI, hold max tip-down
```

`--rot_cmd RX RY RZ` holds a constant normalised rotation action, which is how the sign of each task
rotation axis and the orientation-box limits were verified. In Insert mode the script also prints the
gripper-body heights relative to the board plane at reset and at the end of the run.

> **Silent-freeze trap.** These scripts shrink the PhysX GPU buffers on purpose. The training config
> asks for a 1 GiB collision stack; if another training or play session already holds most of the
> card, `PxgCudaDeviceMemoryAllocator` fails, PhysX cannot launch its narrowphase kernels, and the
> simulation **keeps running while returning the same stale pose every step**. That looks exactly
> like a perfectly-held arm rather than an error. Always check the log for
> `fail to launch kernel` before trusting a measurement taken while something else is on the GPU.

---

## Fixture USD (`env_v7`)

Runtime asset loaded by the scene:

```text
usd_model/env_v7/pcb_insertion_env.usd
```

Regenerate after changing the converter:

```bash
cd usd_model/env_v7 && python3 convert_to_usd.py
```

`env_v7` is the same Onshape assembly as `env_v6`, re-exported with the side rail-guides
(`Part_1_4`, `Part_1_6`) thinned from 15 mm to 4 mm so the gripper jaw/carriage clears them
mid-insert. All materials, friction, and contact/rest offsets are unchanged from v6.

`env_v7` and `usd_model/usd_robot/` are the only USD assets the env loads. Superseded fixtures
(`env_v3` … `env_v6`, `usd_env`) and the archived `weights/` and `video/` directories are
gitignored and kept local only.

### What is included / excluded

| Included | Notes |
|----------|--------|
| Magazine | Triangle-mesh collider (`physics:approximation = none`) |
| Guide rails (`Part_1_7`) | Horizontal PCB support — triangle mesh |
| Side belts (`Part_1_2`) | Triangle mesh |
| Side rail-guides (`Part_1_4`, `Part_1_6`) | Tall posts beside conveyor, thinned to 4 mm (env_v7) — triangle mesh |
| Stand / frame | `convexDecomposition` |

| Excluded | Reason |
|----------|--------|
| Chip / PCB mesh | Spawned as a separate cuboid in the env |
| Short axles (`Part_1_3`) | Decorative |

Contact / rest offsets on fixture meshes default to **0.0001 m**; the PCB cuboid uses the same order
of magnitude in `widowx_pcb_env_cfg.py`.

### Support height vs "floating" look

- The side **belt** top is lower than the guide **rail** tops, so the PCB rests on the **rails**.
- A few mm of gap above the belt in the viewport is normal, not a bug.
- Spawn height comes from `_CONVEYOR_SURFACE_Z` → `_PCB_CENTER_Z_ENV`. Align it with **rail support**,
  not the belt visual.

### PhysX GPU buffers

Full triangle meshes × many envs overflow the collision stack (dropped contacts → tunneling / floaty
PCB), so the sim config asks for:

```text
gpu_collision_stack_size = 2**30   # 1 GiB
```

Prefer **`--num_envs 2048`** (or lower) if overflow returns at 4096. See the silent-freeze warning
above for what happens when this allocation *fails* instead of overflowing.

**Do not** casually switch the magazine to SDF without retuning `sdf_margin` / resolution: a narrow
slot can behave "thicker" and eject the PCB.

---

## TensorBoard

```bash
cd /path/to/widowx_pcb
python agents/monitor_tensorboard.py --port 6006
# or: tensorboard --logdir=logs --port=6006
```

Open <http://127.0.0.1:6006>. Runs: `rl_games/widowx_pcb_approach/summaries` and
`rl_games/widowx_pcb_insert/summaries`.

| Tag | Meaning |
|-----|---------|
| `Episode_Reward/<term>` | Per-term contribution — already `dt`-scaled, so compare terms, not absolute magnitudes |
| `Episode_Termination/approach_success` | Phase 1 success rate |
| `Episode_Termination/insert_success` | Phase 2 success rate |
| `Episode_Reward/failure_penalty` | How often Insert is ending in failure rather than timeout |
| `Episode_Reward/wrist_pitch_deg_debug` | Signed wrist pitch (deg); divide by 1e-10; negative = tip-down |
| `Episode_Reward/jaw_clearance_mm_debug` | Lower-jaw height above rail (mm); divide by 1e-10 |
| `Episode_Reward/board_travel_mm_debug` | Leading-edge +Y travel since reset (mm); divide by 1e-10 |
| `Episode_Reward/board_lane_drift_mm_debug` | Leading-edge X lane drift (mm); divide by 1e-10 |
| `Curriculum/approach_gripper_debug/closedness_tight` | Phase 1 quality on the same σ as the success test |

Sparse terms (e.g. `insert_travel_milestone`) average over envs and are `dt`-scaled, so they can look
near-zero even while firing on a few envs.

---

## Key constants (`widowx_pcb_env_cfg.py`)

| Constant | Role |
|----------|------|
| `_MAG_POS` / `_MAG_ROT_WXYZ` | Fixture placement (−90° about Z maps root −X → world +Y push) |
| `PUSH_AXIS_WORLD` | `(0, 1, 0)` |
| `_ROBOT_BASE_POS` / `_ROBOT_BASE_ROT_WXYZ` | Side placement, identity rotation |
| `_ROBOT_HOME_JOINT_POS` | FK-solved reset posture (see `diag_side_base_home_pose.py`) |
| `_APPROACH_JAW_SPAN_M` | 30 mm open jaw span |
| `_APPROACH_FINGER_OFFSET_M` / `_APPROACH_GAP_*_M` | ±15 mm trailing-edge finger / width-gap targets |
| `_APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD` | Success closedness (0.40, σ = 35 mm) |
| `_APPROACH_SUCCESS_TIP_MID_THICKNESS_THRESHOLD` | Success tip-mid index (0.60) |
| `_APPROACH_SUCCESS_MIN_TIP_DOWN_DEG` / `_APPROACH_WRIST_TARGET_PITCH_DOWN_DEG` | Success pitch gate (17.5°) / shaping target (25°) |
| `_APPROACH_PROXIMITY_STD_M` / `_APPROACH_OVERSHOOT_STD_M` | Far-field close (10 cm) / overshoot brake (8 mm) |
| `_INSERT_SUCCESS_LEAD_Y_ENV` | Leading-edge success terminus along +Y |
| `_INSERT_PUSH_APPROACH_MAX_STEP_M` / `_INSERT_PUSH_REF_SPEED_M_S` | Progress / velocity normalisation (0.0006 m, 0.062 m/s) |
| `_INSERT_PUSH_MIN_TIP_MID` | Tip-mid seating gate for push rewards (0.55) |
| `_INSERT_JAW_RAIL_CLEARANCE_TARGET_M` | Full credit for `jaw_rail_clearance` (28 mm) |
| `_INSERT_TRAVEL_MILESTONE_FRACTIONS` | Milestone tiers (12/25/50/75/87.5/95 %) |
| `_INSERT_ORIENTATION_DEV_LIMITS_PER_AXIS` | Per-axis rotation box for Insert |
| `_APPROACH_STATES_PATH` | Terminal-state buffer consumed by Insert |

Re-align magazine constants (`_MAG_Y_NEAR_FACE_ENV`, success XY) with Isaac Sim measurements whenever
the fixture moves.

---

## Branching model

- **`main`** — release-oriented working line.
- **`dev`** — day-to-day integration.
- **`side_arm`** — current line for the side-mounted base architecture.

---


## License

See `LICENSE` in this package.
