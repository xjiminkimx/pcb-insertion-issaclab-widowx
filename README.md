# WidowX PCB Approach + Slide — Isaac Lab

Manager-based RL task for a **WidowX** arm standing **beside** a conveyor. The task is split into two
independently trained phases, chained through a terminal-state buffer:

1. **Approach** — open-jaw **straddle** of the PCB trailing short edge (no board motion allowed).
2. **Slide** — push the straddled board **+Y** along the guide rails into the magazine slot.

Both phases use **task-space variable impedance control** (Isaac Lab OSC) and are trained with
**rl-games** PPO. Checkpoints and TensorBoard logs land under this package's `logs/`.

| Task ID | Env cfg | Status |
|---------|---------|--------|
| **`Isaac-WidowX-PCB-Approach-v0`** | `WidowXPcbApproachEnvCfg` | Active — phase 1 |
| **`Isaac-WidowX-PCB-Slide-v0`** | `WidowXPcbSlideEnvCfg` | Active — phase 2 |
| `Isaac-WidowX-PCB-Push-v0` | → Approach | Deprecated alias |
| `Isaac-WidowX-PCB-Straddle-v0` | → Approach | Deprecated alias |

Robot USD: `usd_model/usd_robot/wxai/wxai_follower.usd`.

---

## Pipeline

```text
train_approach.sh  ->  collect_approach_states.sh  ->  train_slide.sh
   (phase 1)            data/approach_terminal_states.npz     (phase 2)
```

**Phase 1 — Approach** (`episode_length_s = 3.0`)

1. PCB spawns on the rails with randomized X / yaw; gripper is PD-held open at a **30 mm** span.
2. Arm resets to a fixed FK-solved home posture beside the belt.
3. Policy drives the pads onto the trailing short edge: pads at mid-thickness, jaws straddling the
   board across its **width**, wrist trailing the pads along +Y.
4. Terminates on `approach_success` (closedness **and** tip-mid-thickness above threshold) — any +Y
   board displacement is penalised, sliding is *not* part of this phase.

**Phase 2 — Slide** (`episode_length_s = 8.0`)

1. Reset replays a random Approach terminal state (arm joints + PCB placed relative to the gripper).
2. Policy pushes the trailing edge +Y with the jaws held open, keeping the board flat and on-lane.
3. Terminates on `slide_success` (leading edge seated near the magazine back wall) or on failure.

> The terminal-state buffer is tied to the base placement and the Approach policy that produced it.
> After moving the robot base or retraining Approach, re-run `collect_approach_states.sh` before
> training Slide, or Slide will reset into a pose that no longer exists.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers the Approach / Slide envs (+ deprecated aliases). |
| `widowx_pcb_env_cfg.py` | Scene, geometry constants, per-phase actions / observations / rewards / events / terminations. |
| `mdp_custom.py` | Task-space impedance action term, geometry helpers, reward & termination functions, reset helpers. |
| `usd_model/env_v7/` | `assembly_1.urdf` + `convert_to_usd.py` → **`pcb_insertion_env.usd`** (runtime fixture). |
| `usd_model/usd_robot/wxai/` | WidowX follower USD. |
| `scripts/train_approach.sh` / `play_approach.sh` | Phase 1 train / play. |
| `scripts/train_slide.sh` / `play_slide.sh` | Phase 2 train / play. |
| `scripts/collect_approach_states.py` / `.sh` | Roll out an Approach checkpoint and store terminal states. |
| `scripts/diag_side_base_home_pose.py` | FK search / inspection of the reset posture and base placement. |
| `scripts/diag_ee_box.py` | Instruments the EE translation / rotation boxes; measures droop, wrist pitch, clearance. |
| `agents/` | PPO configs, rl-games patches, TensorBoard helper, workspace paths. |
| `data/` | Terminal-state `.npz` buffer (see `data/README.md`). |
| `logs/rl_games/widowx_pcb_{approach,slide}/` | Summaries + `nn/*.pth` checkpoints. |

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
bash scripts/train_slide.sh --num_envs 2048 --headless
```

Both train scripts accept `--clean-logs` (drop TensorBoard summaries, keep checkpoints) and
`--clean-all` (drop the whole phase log dir).

PPO knobs live in `agents/rl_games_ppo_cfg.py`:

| Config | `name` | `max_epochs` | `horizon_length` | `clip_actions` |
|--------|--------|--------------|------------------|----------------|
| `WidowXPcbApproachPPOCfg` | `widowx_pcb_approach` | 150 | 128 | 0.25 |
| `WidowXPcbSlidePPOCfg` | `widowx_pcb_slide` | 120 | 256 | 0.25 |

`agents/rl_games_logstd_safety.py` is monkey-patched in at import time to stop `Normal.sample` from
failing when `exp(log_std)` underflows.

---

## Evaluate / play

**Important:** Isaac Lab `play.py` resolves `logs/rl_games/...` relative to **cwd** — run from this
package root.

```bash
cd /path/to/widowx_pcb
bash scripts/play_approach.sh --num_envs 1
bash scripts/play_slide.sh --num_envs 16
```

Without `--checkpoint`, play loads the best `nn/<name>.pth`; `--use_last_checkpoint` picks the latest
epoch file; `--real-time` gives wall-clock playback.

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
absolute-from-reset clamps anchored by the `store_reset_ee_pose` / `store_slide_reset_ee_pose` reset
events:

- **`task_position_box_enabled`** — without it, gravity-compensation residual sinks the arm and the
  sunk pose becomes the new setpoint (the "쳐짐" ratchet). Measured with zero actions: pad Z −6 mm at
  0.2 s, −38 mm at 2 s, −71 mm at 3.8 s. With the box the offset stays within a few mm.
- **`task_orientation_box_enabled`** — the same ratchet in rotation. Measured on Slide with the
  position box already on: wrist pitch ran −13° → −25° in 1.6 s and kept falling linearly, i.e. the
  gripper head simply flops down over an episode. **Stiffness cannot fix either one** — the tracking
  error is ~0 by construction, so the impedance spring never generates a restoring term.

| Parameter | Approach | Slide |
|-----------|----------|-------|
| `position_scale` / `orientation_scale` | 0.15 / 0.15 | 0.04 / 0.10 |
| lateral (X) half-range | ±0.08 m | ±0.01 m |
| vertical (Z) half-range | ±0.06 m | ±0.01 m |
| push (Y) offset range | −0.05 … +0.20 m | 0 … +0.60 m |
| rotation bound | ±0.15 rad (≈ ±8.6°), all three axes | per-axis (see below) |

Note that `_ARM_TASK_ORIENTATION_MAX_DEV_RAD = 0.15` is **radians**, i.e. ≈ 8.6°, not the 15° the
inline comment claims. That bound is tight enough that Approach cannot correct the ~8° jaw roll it
inherits from the reset posture, which is why Slide starts with the pads a few mm out of level.

Slide needs an **asymmetric per-axis** rotation bound, because the wrist must be free to pitch tens of
degrees down for rail clearance while roll and yaw stay pinned near the straddle orientation. A
single symmetric scalar cannot express that, hence `orientation_dev_limits_per_axis`:

```python
_SLIDE_ORIENTATION_DEV_LIMITS_PER_AXIS = (
    (-0.30, 0.09),   # rx  pitch: negative = tip-DOWN
    (-0.10, 0.10),   # ry  roll about the push axis
    (-0.12, 0.12),   # rz  yaw about vertical
)
```

Measured effect on Slide: zero-action equilibrium settles at **−18°** (reset −13° plus ~5° of static
gravity sag) and stays there; full tip-down command saturates at **−35°**; full tip-up at −13°.

### Selective compliance (Slide)

`_ARM_TASK_SLIDE_STIFFNESS_LIMITS_PER_AXIS` caps `K` per axis regardless of what the policy asks for:
`ty` (push) is stiffest at 250–2500 so the board actually breaks static friction, `tx`/`rz` are softer
so it can self-align in the slot, and `tz`/`rx`/`ry` are soft but **never free** — a fully
uncontrolled axis has literally zero restoring torque and once `ty` could deliver real force, the
coupling into those axes diverged and crashed PhysX.

---

## Phase 1 — Approach

### Rewards (`RewardsApproachCfg`)

| Term | Weight | Role |
|------|--------|------|
| `trailing_face_approach` | 150 | Bell-shaped approach to the trailing face; the only term that decays on overshoot past the edge |
| `tip_mid_thickness` | 150 | Pull both pads to the PCB mid-thickness plane (not the top/bottom face) |
| `finger_proximity` | 150 | Wide-σ closing signal toward the ±15 mm trailing-edge finger targets |
| `lateral_gap` | 120 | Symmetric jaw-to-board width gaps (fixes left/right asymmetry) |
| `between_fingers` | 60 | Board between the jaws |
| `pcb_forward_push_penalty` | −3000 | Flat penalty once the board moves >5 mm toward the slot — Approach must not slide |
| `approach_success_bonus` | 20000 | Sparse success bonus |
| `action_rate_penalty` | −0.002 | Smooth actions |
| `approach_gripper_debug_monitor` | 1e-10 | Debug accumulation only |

The five dense terms use `*_fade_near_success` wrappers: their credit fades out as tight closedness
approaches the success bar, so farming a full episode of shaping cannot beat terminating early with
the success bonus.

### Success and terminations

`approach_success` requires **both** `closedness_tight ≥ 0.55` (σ = 25 mm) **and**
`tip_mid_thickness ≥ 0.3`. Failures: `pcb_tilt_excessive`, `pcb_long_axis_not_horizontal`,
`pcb_fallen_below_rail`, `pcb_moving_backward`, plus `time_out`.

### Debug scalars

`Curriculum/approach_gripper_debug/*` — `closedness` (loose σ) vs `closedness_tight` (the σ the
success test actually uses), `closedness_peak`, `straddle_success_frac`, per-side gaps in mm. Do not
read the loose `closedness` as a success predictor; only `closedness_tight` shares the success σ.

---

## Phase 2 — Slide

### Rewards (`RewardsSlideCfg`)

| Term | Weight | Role |
|------|--------|------|
| `leading_edge_push_progress` | 50 | Δ(+Y) of the leading edge, ungated |
| `push_axis_velocity` | 50 | +Y board velocity, zeroed if it skids/lifts off-axis |
| `slide_travel_milestone` | 50 | One-shot bonuses at 25/50/75/87.5/95 % of travel |
| `goal_lead_proximity` | 10 | Mild terminal-approach shaping |
| `straddle_hold` | 10 | Keep the grip pointed at the trailing edge |
| `pcb_yaw_alignment` | 5 | Board yaw aligned with the push axis |
| `lateral_gap` | 4 | Keep the board centred between the jaws |
| `wrist_tip_down` | 8 | Monotone ramp on tip-down wrist pitch (rail clearance posture) |
| `alive_penalty` | −25 | Makes idling net-negative |
| `failure_penalty` | −10000 | One-shot, on non-timeout non-success terminations |
| `slide_success_bonus` | 25000 | Sparse success bonus |
| `wrist_pitch_deg_debug` | 1e-10 | Logs the signed wrist pitch in degrees to TensorBoard |

**Reward weights are scaled by `step_dt` (0.008 s).** A one-shot event with weight `W` contributes
only `0.008·W` to the return, which is why the sparse bonuses are in the tens of thousands while the
dense terms are single- or double-digit. Sizing them against each other without that factor is the
single easiest way to get this phase wrong. Reference point, measured with zero actions: static
income is ≈18.5 /s, so over the 8 s episode idling collects ≈148 against 200 of alive penalty.

`wrist_tip_down` is a **monotone ramp**, not a peak: zero credit while flat or tip-up, rising linearly
in `sin` space to full credit at `_SLIDE_WRIST_TARGET_PITCH_DOWN_DEG`, held until
`_SLIDE_WRIST_MAX_PITCH_DOWN_DEG`, then decaying. The earlier narrow-Gaussian version evaluated to
~1 % of full credit at the poses Slide actually visits — a flat dead zone the policy could not climb.

> **Open question, measured but unresolved:** deeper tip-down moves the gripper/carriage bodies
> *down* by ≈1.4 mm per degree relative to the board plane (at −13° they sit +22 / +7.6 mm above it;
> at −35°, −7.9 / −22.8 mm below), which is the opposite of the rail-clearance intent the term was
> written for. Confirm visually which part actually fouls the rail guide before trusting the target
> angle. The knobs are the reward target angle and the `rx` bound in the orientation box.

### Success and terminations

`slide_success` needs the leading-edge centre inside the success box near the magazine back wall
with the jaws still closed enough. Failures: `pcb_fallen_below_rail`,
`pcb_long_axis_not_horizontal`, plus `time_out`.

---

## Diagnostics

Both diagnostic scripts are FK / instrumentation tools, not training entry points.

```bash
# Reset-posture and base-placement geometry (jaw axis vs PCB width, pad levelness, reach, joint margins)
python -u scripts/diag_side_base_home_pose.py --inspect --num_envs 4 --headless
python -u scripts/diag_side_base_home_pose.py --num_envs 4096 --headless      # search a new home pose

# EE boxes: is the clamp running, what is it anchored to, how much does the arm droop / flop?
python -u scripts/diag_ee_box.py --num_envs 4 --steps 200 --headless           # Approach
python -u scripts/diag_ee_box.py --slide --num_envs 4 --steps 400 --headless   # Slide + wrist pitch + clearance
python -u scripts/diag_ee_box.py --slide --num_envs 1 --steps 5000 --rot_cmd -1 0 0   # GUI, hold max tip-down
```

`--rot_cmd RX RY RZ` holds a constant normalised rotation action, which is how the sign of each task
rotation axis and the orientation-box limits were verified. In Slide mode the script also prints the
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
mid-slide. All materials, friction, and contact/rest offsets are unchanged from v6.

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
`rl_games/widowx_pcb_slide/summaries`.

| Tag | Meaning |
|-----|---------|
| `Episode_Reward/<term>` | Per-term contribution — already `dt`-scaled, so compare terms, not absolute magnitudes |
| `Episode_Termination/approach_success` | Phase 1 success rate |
| `Episode_Termination/slide_success` | Phase 2 success rate |
| `Episode_Reward/failure_penalty` | How often Slide is ending in failure rather than timeout |
| `Episode_Reward/wrist_pitch_deg_debug` | Achieved signed wrist pitch (negative = tip-down) |
| `Curriculum/approach_gripper_debug/closedness_tight` | Phase 1 quality on the same σ as the success test |

Sparse terms (e.g. `slide_travel_milestone`) average over envs and are `dt`-scaled, so they can look
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
| `_APPROACH_FINGER_OFFSET_M` | ±15 mm trailing-edge finger targets |
| `_APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD` / `_APPROACH_SUCCESS_TIP_MID_THICKNESS_THRESHOLD` | Success gates (0.55 / 0.3) |
| `_SLIDE_SUCCESS_LEAD_Y_ENV` | Leading-edge success terminus along +Y |
| `_SLIDE_WRIST_TARGET_PITCH_DOWN_DEG` / `_SLIDE_WRIST_MAX_PITCH_DOWN_DEG` | Tip-down ramp shape |
| `_SLIDE_ORIENTATION_DEV_LIMITS_PER_AXIS` | Per-axis rotation box for Slide |
| `_APPROACH_STATES_PATH` | Terminal-state buffer consumed by Slide |

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
