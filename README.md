# PCB on-rail task — WidowX (Isaac Lab)

This package is an **Isaac Lab** manager-based RL task: a robot arm interacts with a **PCB** on a **magazine + conveyor** assembly (single aligned USD). Side guide-rail rods are omitted from the fixture USD. Training uses **rl-games** (PPO).

Two simulation variants are registered:

| Robot | Task ID | Env config |
|--------|---------|------------|
| **WidowX** (Trossen `usd_model/usd_robot/wxai/wxai_follower.usd`) | `Isaac-WidowX-PCB-Grasp-v0` | Phase 1 — grasp trailing short-edge centre |
| | `Isaac-WidowX-PCB-Insert-v0` | Phase 2 — insert grasped PCB along +Y into slot |

Robot runtime assets now live under `usd_model/usd_robot/`, and environment fixtures plus conversion sources live under `usd_model/usd_env/`.

**High-level episode flow (two-phase curriculum with policy chaining):**

1. **Phase 1 — Grasp** (`Isaac-WidowX-PCB-Grasp-v0`): PCB on the conveyor, gripper open. Rewards shape approach, straddle, and tight closure on the trailing short-edge centre. The episode ends on successful grasp or timeout.
2. **Collect terminal states**: Roll out the trained Grasp policy and save robot + PCB poses at every successful grasp termination into `data/grasp_terminal_states.npz`.
3. **Phase 2 — Insert** (`Isaac-WidowX-PCB-Insert-v0`): Each episode reset **samples** a saved Grasp terminal state (robot joints + PCB pose). A dense SDF-style reward shapes motion of the leading edge toward the first magazine slot along +Y.

Recommended training order: **Grasp → collect states → Insert**.

See [Policy chaining (Grasp → Insert)](#policy-chaining-grasp--insert) for the full workflow and file formats.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers grasp/insert envs and applies the rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | WidowX scene, actions, observations, rewards, events, terminations, magazine/rail geometry. |
| `usd_model/env_v3/` | URDF + meshes + `convert_to_usd.py` → `usd_env/pcb_insertion_env.usd` (short axle rods / width cross-rods omitted) |
| `usd_model/usd_robot/` | Robot USD bundles (wxai follower). |
| `mdp_custom.py` | Custom MDP terms (grasp/insert rewards, rail reset, policy-chaining reset, SDF insertion reward). |
| `scripts/collect_grasp_states.py` | Roll out Grasp policy; save successful terminal states to `.npz`. |
| `scripts/train_grasp.sh` / `scripts/train_insert.sh` | Workspace training entry points (logs under `logs/`). |
| `scripts/play_grasp.sh` / `scripts/play_insert.sh` | Evaluate checkpoints from workspace `logs/rl_games/`. |
| `scripts/record_insert_videos.py` / `scripts/record_insert_videos.sh` | Record Insert rollout videos (tuned camera, fast headless capture). |
| `data/grasp_terminal_states.npz` | Grasp terminal-state buffer (created by the collection script; required for Insert training). |
| `agents/` | PPO configs (`WidowXPcbGraspPPOCfg` / `WidowXPcbInsertPPOCfg`), log-std safety helper, TensorBoard monitor script. |

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

## Train (rl-games PPO)

**Recommended:** use the workspace scripts so TensorBoard logs and checkpoints are saved under **`logs/` in this repo**.

**Phase 1 — grasp only:**

```bash
conda activate isaac-sim   # or your env name
cd /path/to/widowx_pcb   # this workspace
bash scripts/train_grasp.sh --num_envs 4096 --headless
```

**Phase 2 — insert only** (requires `data/grasp_terminal_states.npz`; see [Policy chaining](#policy-chaining-grasp--insert)):

```bash
bash scripts/train_insert.sh --num_envs 4096 --headless
```

If you already trained from the Isaac Lab root, copy logs into the workspace once:

```bash
bash scripts/sync_logs_from_isaaclab.sh
```

Alternative (logs under `<IsaacLab>/logs/` instead of this workspace):

```bash
cd /path/to/IsaacLab
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-Grasp-v0 --num_envs 4096 --headless
```

---

## Policy chaining (Grasp → Insert)

This task trains two separate policies and chains them using ideas from **Sequential Dexterity** ([Chen et al., CoRL 2023](https://arxiv.org/abs/2309.00987)): the *terminal state distribution* of Phase 1 becomes the *initial state distribution* of Phase 2. Insert-phase rewards use an SDF-inspired dense shaping term inspired by **IndustReal** ([Tang et al., RSS 2023](https://arxiv.org/abs/2305.17110)).

### Why chain policies?

Training Insert from a fixed “ideal” grasp pose (e.g. kinematic snap to closed jaws) creates a **distribution mismatch**: the Insert policy never sees the small pose errors, gripper gaps, and PCB offsets that a real Grasp policy produces. Sampling resets from recorded Grasp successes exposes Insert training to that same distribution and improves transfer when the two policies are run back-to-back.

### End-to-end workflow

```text
  ┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
  │  Train Grasp    │────▶│  collect_grasp_states.py │────▶│  Train Insert   │
  │  (Phase 1 RL)   │     │  → grasp_terminal_states │     │  (Phase 2 RL)   │
  └─────────────────┘     └──────────────────────────┘     └─────────────────┘
```

**Step 1 — Train the Grasp policy**

```bash
cd /path/to/widowx_pcb
bash scripts/train_grasp.sh --num_envs 2048 --headless
```

Checkpoint example: `logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth`

**Step 2 — Collect successful terminal states**

Roll out the Grasp checkpoint in parallel envs. Whenever an episode terminates on **`grasp_success`** (not timeout), the script records the robot joint positions and PCB root pose at that step.

```bash
python scripts/collect_grasp_states.py \
    --checkpoint logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth \
    --num_envs 256 \
    --num_states 2000 \
    --out data/grasp_terminal_states.npz \
    --headless
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--checkpoint` | *(required)* | Path to the trained Grasp `.pth` |
| `--num_envs` | `256` | Parallel rollout environments |
| `--num_states` | `2000` | Target count of **successful** terminal states to save |
| `--out` | `data/grasp_terminal_states.npz` | Output file |
| `--max_steps` | `20000` | Safety cap on total env steps |

`--num_states` counts individual successful grasp terminations (not full episodes). With many parallel envs, several successes can occur in the same step, so collection finishes faster than running `--num_states` sequential episodes.

**Step 3 — Train the Insert policy**

`scripts/train_insert.sh` checks that `data/grasp_terminal_states.npz` exists, then launches Insert training. On every episode reset, `reset_from_grasp_states` samples one row from the buffer and writes it to the robot and PCB.

```bash
bash scripts/train_insert.sh --num_envs 2048 --headless
```

### Saved state format (`grasp_terminal_states.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `joint_pos` | `(N, n_joints)` | Robot joint positions at grasp success (rad / m) |
| `pcb_pos_env` | `(N, 3)` | PCB root position in **env-local** coordinates (m) |
| `pcb_quat` | `(N, 4)` | PCB root orientation quaternion **(w, x, y, z)** |
| `joint_names` | list of str | Joint name for each column of `joint_pos` |

`N` is the number of collected successes (≤ `--num_states`). The Insert reset maps `joint_names` onto the live articulation by name, so column order stays consistent across runs.

### How Insert reset uses the buffer

Implemented in `mdp_custom.reset_from_grasp_states`, wired as `EventCfgInsert.reset_from_grasp`:

1. Load `data/grasp_terminal_states.npz` once (cached for the process).
2. For each env being reset, draw a random index with replacement.
3. Write sampled `joint_pos` to the robot (clamped to soft limits, zero velocity).
4. Write sampled `pcb_pos_env` + `pcb_quat` to the PCB rigid body (env-local pos + `env_origins` → world).

Path constant in config: `_GRASP_STATES_PATH` in `widowx_pcb_env_cfg.py`.

### Insert-phase reward (SDF-inspired)

Phase 2 uses `pcb_insertion_sdf_reward` (`RewardsInsertPhaseCfg.sdf_insert`, weight 80). It combines three terms in `[0, 1]` (no USD SDF queries — purely tensor geometry, following the *spirit* of IndustReal’s signed-distance shaping):

| Component | Coef | Signal |
|-----------|------|--------|
| **Proximity** | 0.20 | Gaussian on 3D distance from PCB **leading edge** to slot centre (`exp(-dist / σ)`) |
| **Alignment** | 0.20 | \|cos θ\| between PCB long axis and world +Y (push direction) |
| **Depth** | 0.60 | Linear fraction of leading-edge penetration past the slot mouth |

Supporting terms: velocity toward the slot (`insert_y_toward_slot`) and lateral-slide penalty.

Slot geometry constants (tune in Isaac Sim after loading the scene):

- `_SLOT_CENTER_XYZ_ENV` — env-local centre of the first slot opening
- `_SLOT_DEPTH_M`, `_SLOT_HALF_DIMS_XYZ` — slot box half-extents
- `_SLOT_MOUTH_Y_ENV` — slot entrance along +Y

### Grasp success criterion (Phase 1)

Grasp bonus and episode termination fire when `grasp_edge_center_achieved` is true: straddle + edge proximity + centre alignment + pinch readiness, and gripper gap `left_carriage_joint < PCB_Z × 1.1`.

### Tips

- Collect **more states than you have parallel Insert envs** (e.g. 2000+ states for 2048 envs) so resets stay diverse.
- If Insert training fails immediately with `FileNotFoundError`, run Step 2 first.
- Re-collect the buffer whenever you change Grasp checkpoints, success criteria, or domain randomization — the Insert initial distribution should match the Grasp policy you will deploy.
- After moving the magazine in sim, re-tune `_SLOT_CENTER_XYZ_ENV` and related constants before Insert training.

---

## Evaluate / play a trained policy

Checkpoints live under this workspace (when you train with `scripts/train_*.sh`):

| Phase | Log folder | Default checkpoint |
|-------|------------|-------------------|
| Grasp | `logs/rl_games/widowx_pcb_grasp/` | `nn/widowx_pcb_grasp.pth` |
| Insert | `logs/rl_games/widowx_pcb_insert/` | `nn/widowx_pcb_insert.pth` |

**Important:** Isaac Lab `play.py` resolves `logs/rl_games/...` relative to the **current working directory**. Run from **`/path/to/widowx_pcb`** (this workspace), not from the Isaac Lab repo root — otherwise it will not find workspace checkpoints.

### Grasp policy (recommended)

```bash
cd /path/to/widowx_pcb
bash scripts/play_grasp.sh --num_envs 16
```

### Insert policy (recommended)

```bash
cd /path/to/widowx_pcb
bash scripts/play_insert.sh --num_envs 16
```

Without `--checkpoint`, `play.py` auto-loads the best model from `logs/rl_games/widowx_pcb_<phase>/nn/<phase>.pth`.
Use `--use_last_checkpoint` for the most recent epoch file instead of the best one.
Add `--real-time` to throttle stepping to wall clock during interactive play.

### Load a specific checkpoint

```bash
cd /path/to/widowx_pcb
bash scripts/play_grasp.sh \
    --num_envs 16 \
    --checkpoint logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth
```

Insert example:

```bash
bash scripts/play_insert.sh \
    --num_envs 16 \
    --checkpoint logs/rl_games/widowx_pcb_insert/nn/widowx_pcb_insert.pth
```

### Record a video of the rollout

#### Insert policy (recommended — `record_insert_videos`)

For Insert rollouts, use the workspace script instead of `play.py --video`. It loads the latest best checkpoint, uses a **playback camera** that frames the robot arm + PCB + slot (set in `WidowXPcbInsertEnvCfg.viewer`), and records headless with faster defaults (854×480, one frame every 4 sim steps, 30 fps output).

**Prerequisites:** `conda activate isaac-sim` (or your Isaac Lab env). **Pause Insert training** while recording — a second Isaac Sim on the same GPU is very slow.

```bash
cd /path/to/widowx_pcb
bash scripts/record_insert_videos.sh
```

Default output: `logs/rl_games/widowx_pcb_insert/videos/play/insert-episode-<HHMMSS>.mp4`

**Custom output path** (e.g. workspace root):

```bash
bash scripts/record_insert_videos.sh --out insert-episode-latest.mp4
```

**Specific checkpoint:**

```bash
bash scripts/record_insert_videos.sh \
    --checkpoint logs/rl_games/widowx_pcb_insert/nn/widowx_pcb_insert.pth \
    --out insert-episode-latest.mp4
```

**Most recent epoch file** (not best):

```bash
bash scripts/record_insert_videos.sh --use_last_checkpoint
```

**Python equivalent:**

```bash
python scripts/record_insert_videos.py \
    --headless \
    --num_episodes 1 \
    --checkpoint logs/rl_games/widowx_pcb_insert/nn/widowx_pcb_insert.pth \
    --out insert-episode-latest.mp4
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--checkpoint` | `logs/.../nn/widowx_pcb_insert.pth` | Trained Insert `.pth` |
| `--num_episodes` | `1` | Episodes to record |
| `--out` | *(auto)* | Output `.mp4` path |
| `--use_last_checkpoint` | off | Use latest `last_*.pth` instead of best |
| `--video_fps` | `30` | Playback frame rate (readable speed) |
| `--render_stride` | `4` | Capture every N env steps (higher = faster recording) |
| `--video_width` / `--video_height` | `854` / `480` | Render resolution (lower = faster) |
| `--seed` | `42` | Reset seed (change for a different rollout) |

Faster capture (lower quality):

```bash
bash scripts/record_insert_videos.sh \
    --render_stride 6 \
    --video_width 640 \
    --video_height 360
```

Episode length is up to 6 s (~750 env steps). The video ends when the episode terminates (detach, tilt, success, etc.) or hits the timeout.

#### Grasp policy (`play.py --video`)

```bash
cd /path/to/widowx_pcb
bash scripts/play_grasp.sh \
    --num_envs 4 \
    --headless \
    --video \
    --video_length 500
```

Videos are saved under `logs/rl_games/widowx_pcb_<phase>/videos/play/`.

To fix an already-recorded fast video (sim-tagged 125 fps):

```bash
ffmpeg -y -i input.mp4 -vf "setpts=PTS*(125/30)" -r 30 -c:v libx264 -crf 18 -pix_fmt yuv420p output_realtime.mp4
```

---

## TensorBoard

Training with rl-games **automatically** writes TensorBoard event files. No extra flags are required — `use_diagnostics: True` is already set in [`agents/rl_games_ppo_cfg.py`](agents/rl_games_ppo_cfg.py).

When you use `scripts/train_grasp.sh` / `scripts/train_insert.sh`, logs are written under **this workspace**:

| Task | Task ID | Log folder (under workspace) |
|------|---------|------------------------------|
| Grasp | `Isaac-WidowX-PCB-Grasp-v0` | `logs/rl_games/widowx_pcb_grasp/` |
| Insert | `Isaac-WidowX-PCB-Insert-v0` | `logs/rl_games/widowx_pcb_insert/` |

TensorBoard event files live in each run's `summaries/` subdirectory, e.g.:

`logs/rl_games/widowx_pcb_grasp/summaries/events.out.tfevents.*`

### View logs

**Option A — project helper** (defaults to workspace `logs/`):

```bash
cd /path/to/widowx_pcb
python agents/monitor_tensorboard.py --port 6006
```

**Option B — Isaac Lab wrapper** (if logs are still under Isaac Lab root):

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs --port=6006
```

Open <http://127.0.0.1:6006>. In the run selector, pick `widowx_pcb_grasp` or `widowx_pcb_insert`.

### What to watch

- Episodic reward / `rewards/episode_rewards` (or similar rl-games tag)
- Policy / actor loss
- Value / critic loss
- Entropy
- KL / `approx_kl`

### Logging frequency

rl-games logs episode-level metrics when episodes **terminate**. With `horizon_length: 128` and `episode_length_s` of 8–12 s, scalar updates may appear every ~15–60 epochs. That is expected rl-games behavior, not a missing-log bug.

---

## Important task knobs

### Fixture pose vs USD

The runtime fixture is `usd_model/usd_env/pcb_insertion_env.usd`, generated from **env_v3** (`assembly_2.urdf`):

```bash
cd usd_model/env_v3 && python3 convert_to_usd.py
```

Align `_MAG_POS` / `_MAG_ROT_WXYZ` in `widowx_pcb_env_cfg.py` with the loaded asset in world frame. If the **green PCB clips into the conveyor**, raise spawn Z (or `collision_props.contact_offset` / `rest_offset`) until the board sits on the belt top in the contact view.

### Physics and collision

If the PCB intersects the fixture: tune CCD and contact offsets in the env config and collision meshes in USD.

### Reset pose and drop termination

Tune **`pcb_tilt_excessive`**, **`pcb_long_axis_not_horizontal`**, **`pcb_fallen_below_rail`**, **`pcb_dropped`**, and **`arm_idle`** (`_ARM_IDLE_MIN_STEPS`, `_ARM_IDLE_MAX_ABS_VEL_RAD_S`) if episodes reset too aggressively—or not enough when the PCB slips / the policy freezes.

### Grippers and observations

- **WidowX:** fingertip bodies `gripper_left` / `gripper_right`, drive joint `left_carriage_joint`.

### If mean reward plateaus (policy idles near the edge)

1. In grasp/insert reward configs, balance approach / insertion terms vs **`action_rate_penalty`**.
2. In PPO (`agents/rl_games_ppo_cfg.py`), raise **entropy** slightly or decay it more slowly if the policy collapses early.
3. Log **per-term rewards** in TensorBoard if available, to see which term is flat.
