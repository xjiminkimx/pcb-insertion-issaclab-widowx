# PCB on-rail task — WidowX (Isaac Lab)

This package is an **Isaac Lab** manager-based RL task: a robot arm interacts with a **PCB** on a **magazine + conveyor** assembly (single aligned USD). Side guide-rail rods are omitted from the fixture USD. Training uses **rl-games** (PPO).

Three simulation variants are registered (policy chaining):

| Robot | Task ID | Phase |
|--------|---------|--------|
| **WidowX** (`usd_model/usd_robot/wxai/wxai_follower.usd`) | `Isaac-WidowX-PCB-Grasp-v0` | 1 — grasp trailing short-edge centre |
| | `Isaac-WidowX-PCB-Slide-v0` | 2 — slide on guide rails to slot **mouth** |
| | `Isaac-WidowX-PCB-Insert-v0` | 3 — insert through mouth into magazine (SDF reward) |

Robot runtime assets live under `usd_model/usd_robot/`; fixtures under `usd_model/usd_env/`.

**High-level pipeline:**

1. **Grasp** — open gripper, approach, pinch trailing edge (`grasp_success`).
2. **Collect** → `data/grasp_terminal_states.npz`
3. **Slide** — reset from grasp buffer; +Y rail push until leading edge reaches slot mouth (`slide_success`). **No SDF insert reward.**
4. **Collect** → `data/slide_terminal_states.npz`
5. **Insert** — reset from slide buffer; SDF-shaped reward seats PCB in magazine (`insert_success`).

Recommended order: **Grasp → collect → Slide → collect → Insert**.

See [Policy chaining (Grasp → Slide → Insert)](#policy-chaining-grasp--slide--insert).

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers grasp / slide / insert envs; rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | Scene, per-phase rewards/events/terminations, magazine geometry. |
| `usd_model/env_v3/` | URDF + `convert_to_usd.py` → `usd_env/pcb_insertion_env.usd` |
| `mdp_custom.py` | Grasp/slide/insert MDP terms, terminal-state resets, `pcb_insertion_sdf_reward`. |
| `scripts/collect_grasp_states.py` | Grasp → `data/grasp_terminal_states.npz` |
| `scripts/collect_slide_states.py` | Slide → `data/slide_terminal_states.npz` |
| `scripts/train_grasp.sh` / `train_slide.sh` / `train_insert.sh` | Training entry points (`logs/rl_games/`). |
| `scripts/play_grasp.sh` / `play_slide.sh` / `play_insert.sh` | Play trained checkpoints. |
| `scripts/record_insert_videos.py` | Record Insert rollout videos. |
| `data/*.npz` | Terminal-state buffers for policy chaining (see `data/README.md`). |
| `agents/` | `WidowXPcbGraspPPOCfg`, `WidowXPcbSlidePPOCfg`, `WidowXPcbInsertPPOCfg`. |

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

**Phase 2 — slide** (requires `data/grasp_terminal_states.npz`):

```bash
bash scripts/train_slide.sh --num_envs 4096 --headless
```

**Phase 3 — insert** (requires `data/slide_terminal_states.npz`; see [Policy chaining](#policy-chaining-grasp--slide--insert)):

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

## Policy chaining (Grasp → Slide → Insert)

Three policies are trained and chained using **Sequential Dexterity** ([Chen et al., CoRL 2023](https://arxiv.org/abs/2309.00987)): each phase’s terminal state distribution becomes the next phase’s reset distribution.

**Slide** and **Insert** are split because contact at the slot mouth differs from deep magazine insertion — slide learns open-rail +Y pushing and alignment; insert learns SDF-shaped seating inside the slot ([IndustReal](https://arxiv.org/abs/2305.17110)-style dense reward).

### End-to-end workflow

```text
  Train Grasp ──▶ collect_grasp_states ──▶ Train Slide ──▶ collect_slide_states ──▶ Train Insert
                  grasp_terminal_states              slide_terminal_states
```

**Step 1 — Grasp**

```bash
bash scripts/train_grasp.sh --num_envs 2048 --headless
```

**Step 2 — Collect grasp terminal states** (`grasp_success` only)

```bash
python scripts/collect_grasp_states.py \
    --checkpoint logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth \
    --num_envs 256 --num_states 2000 --headless
```

**Step 3 — Slide** (resets from `data/grasp_terminal_states.npz`)

```bash
bash scripts/train_slide.sh --num_envs 2048 --headless
```

Slide rewards (`RewardsSlidePhaseCfg`): straddle hold, +Y step/state progress toward **`_MAG_Y_NEAR_FACE_ENV`** (slot mouth plane), rail-parallel push, lane penalties. **No `sdf_insert`.** Episode ends on **`slide_success`** when the leading edge reaches the mouth.

**Step 4 — Collect slide terminal states** (`slide_success` only)

```bash
python scripts/collect_slide_states.py \
    --checkpoint logs/rl_games/widowx_pcb_slide/nn/widowx_pcb_slide.pth \
    --num_envs 256 --num_states 2000 --headless
```

**Step 5 — Insert** (resets from `data/slide_terminal_states.npz`)

```bash
bash scripts/train_insert.sh --num_envs 2048 --headless
```

### Terminal state format (both `.npz` files)

| Key | Shape | Description |
|-----|-------|-------------|
| `joint_pos` | `(N, n_joints)` | Robot joints at phase success |
| `pcb_pos_env` | `(N, 3)` | PCB root position (env-local, m) |
| `pcb_quat` | `(N, 4)` | PCB orientation **(w, x, y, z)** |
| `joint_names` | list | Joint names matching `joint_pos` columns |

Reset helpers: `reset_from_grasp_states` + `reset_pcb_from_grasp_states` (same code path for slide buffer; path constants `_GRASP_STATES_PATH` / `_SLIDE_STATES_PATH`).

### Insert-phase reward (SDF-inspired)

`RewardsInsertPhaseCfg.sdf_insert` → `pcb_insertion_sdf_reward` (weight 80):

| Component | Coef | Signal |
|-----------|------|--------|
| Proximity | 0.20 | Gaussian on leading-edge distance to slot centre |
| Alignment | 0.20 | \|cos θ\| long axis vs +Y |
| Depth | 0.60 | Leading-edge penetration past mouth (`_MAG_Y_NEAR_FACE_ENV`) |

Also: straddle hold, seated leading-edge proximity, lane / lateral / Z-lift penalties.

### Phase success criteria

| Phase | Termination | Criterion |
|-------|-------------|-----------|
| Grasp | `grasp_success` | Edge-centre pinch + tight gripper |
| Slide | `slide_success` | Mouth + flat/align + **gripper closed** + straddle |
| Insert | `insert_success` | PCB centre Y at magazine centre |

### Tips

- Collect **more states than parallel envs** (e.g. 2000+ for 2048 envs).
- Re-collect buffers when upstream checkpoints or success criteria change.
- Tune `_MAG_Y_NEAR_FACE_ENV`, `_SLOT_CENTER_XYZ_ENV` in Isaac Sim after moving the fixture.

---

## Evaluate / play a trained policy

Checkpoints live under this workspace (when you train with `scripts/train_*.sh`):

| Phase | Log folder | Default checkpoint |
|-------|------------|-------------------|
| Grasp | `logs/rl_games/widowx_pcb_grasp/` | `nn/widowx_pcb_grasp.pth` |
| Slide | `logs/rl_games/widowx_pcb_slide/` | `nn/widowx_pcb_slide.pth` |
| Insert | `logs/rl_games/widowx_pcb_insert/` | `nn/widowx_pcb_insert.pth` |

**Important:** Isaac Lab `play.py` resolves `logs/rl_games/...` relative to the **current working directory**. Run from **`/path/to/widowx_pcb`** (this workspace), not from the Isaac Lab repo root — otherwise it will not find workspace checkpoints.

### Grasp policy (recommended)

```bash
cd /path/to/widowx_pcb
bash scripts/play_grasp.sh --num_envs 16
```

### Slide policy (recommended)

```bash
cd /path/to/widowx_pcb
bash scripts/play_slide.sh --num_envs 16
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

Training with rl-games **automatically** writes TensorBoard event files under each phase's `summaries/` folder. No extra CLI flags are required.

PPO settings (experiment name, `max_epochs`, `horizon_length`, `use_diagnostics`, `reward_shaper`) live in [`agents/rl_games_ppo_cfg.py`](agents/rl_games_ppo_cfg.py): `WidowXPcbGraspPPOCfg`, `WidowXPcbSlidePPOCfg`, `WidowXPcbInsertPPOCfg`.

When you use `scripts/train_grasp.sh` / `train_slide.sh` / `train_insert.sh`, logs are written under **this workspace**:

| Phase | Task ID | Train script | Log folder | Checkpoint | `episode_length_s` | `max_epochs` |
|-------|---------|--------------|------------|------------|--------------------|--------------|
| 1 Grasp | `Isaac-WidowX-PCB-Grasp-v0` | `train_grasp.sh` | `logs/rl_games/widowx_pcb_grasp/` | `nn/widowx_pcb_grasp.pth` | 4 s | 500 |
| 2 Slide | `Isaac-WidowX-PCB-Slide-v0` | `train_slide.sh` | `logs/rl_games/widowx_pcb_slide/` | `nn/widowx_pcb_slide.pth` | 8 s | 200 |
| 3 Insert | `Isaac-WidowX-PCB-Insert-v0` | `train_insert.sh` | `logs/rl_games/widowx_pcb_insert/` | `nn/widowx_pcb_insert.pth` | 5 s | 200 |

TensorBoard event files:

```text
logs/rl_games/widowx_pcb_grasp/summaries/events.out.tfevents.*
logs/rl_games/widowx_pcb_slide/summaries/events.out.tfevents.*
logs/rl_games/widowx_pcb_insert/summaries/events.out.tfevents.*
```

Checkpoints and optional play videos sit beside `summaries/` in the same phase folder (`nn/`, `videos/play/`).

**Diagnostics:** Grasp uses `use_diagnostics: True` (full rl-games loss / KL scalars). Slide and Insert set `use_diagnostics: False` to avoid early `explained_variance` NaN spam in TensorBoard; policy / value / entropy tags still appear.

### View logs

**Option A — project helper** (defaults to workspace `logs/`, lists which phase folders exist):

```bash
cd /path/to/widowx_pcb
python agents/monitor_tensorboard.py --port 6006
```

**Option B — Isaac Lab wrapper** (if logs are still under Isaac Lab root):

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs --port=6006
```

**Option C — TensorBoard directly** on this workspace:

```bash
cd /path/to/widowx_pcb
tensorboard --logdir=logs --port=6006
```

Open <http://127.0.0.1:6006>. In the run selector you should see all trained phases, e.g. `widowx_pcb_grasp`, `widowx_pcb_slide`, `widowx_pcb_insert` (under `rl_games/…/summaries`).

### What to watch (all phases)

rl-games training scalars (names vary slightly by version):

- Episodic reward / `rewards/episode_rewards` (or `episode_rewards`)
- Policy / actor loss
- Value / critic loss
- Entropy
- KL / `approx_kl` (Grasp only when `use_diagnostics: True`)

Isaac Lab manager env extras (when logged per episode):

- `Episode_Reward/<term>` — per reward term from `RewardsGraspPhaseCfg`, `RewardsSlidePhaseCfg`, or `RewardsInsertPhaseCfg`
- `Episode_Termination/<term>` — success rate for `grasp_success`, `slide_success`, or `insert_success`

### Phase-specific signals

| Phase | Success termination | Reward terms worth watching |
|-------|---------------------|-----------------------------|
| Grasp | `grasp_success` | `grasp_success_bonus`, `pcb_between_fingers`, `gripper_closing`, `premature_close` |
| Slide | `slide_success` | `slide_success_bonus`, `push_axis_step_progress`, `push_axis_state`, `mouth_approach_proximity`, `rail_parallel_push_progress` (no `sdf_insert`) |
| Insert | `insert_success` | `sdf_insert`, `mouth_approach_proximity`, lane penalties (`pcb_x_lane_escape`, `pcb_lateral_velocity`) |

If mean reward plateaus, compare per-term `Episode_Reward/*` curves against `action_rate_penalty` — a flat success termination while penalties dominate usually means the policy is idling or fighting contact at the slot mouth.

### Logging frequency

rl-games logs episode-level metrics when episodes **terminate**. With `horizon_length` 128 (Grasp) or 256 (Slide / Insert) and `episode_length_s` of 4–8 s, scalar updates may appear every ~10–60 epochs. That is expected rl-games behavior, not a missing-log bug.

### Clean summaries only (keep checkpoints)

Slide and Insert train scripts accept:

```bash
bash scripts/train_slide.sh --clean-logs    # deletes summaries/ only
bash scripts/train_insert.sh --clean-logs
```

Use `--clean-all` on those scripts to wipe the entire phase log folder (`summaries/`, `nn/`, videos). For Grasp, remove `logs/rl_games/widowx_pcb_grasp/summaries/` manually or re-run with a fresh log dir.

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
