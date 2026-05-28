# PCB on-rail task — WidowX (Isaac Lab)

This package is an **Isaac Lab** manager-based RL task: a robot arm interacts with a **PCB** on a **magazine + conveyor** assembly (single aligned USD). Side guide-rail rods are omitted from the fixture USD. Training uses **rl-games** (PPO).

Two simulation variants are registered:

| Robot | Task ID | Env config |
|--------|---------|------------|
| **WidowX** (Trossen `usd_model/usd_robot/wxai/wxai_follower.usd`) | `Isaac-WidowX-PCB-Grasp-v0` | Phase 1 — grasp trailing short-edge centre |
| | `Isaac-WidowX-PCB-Insert-v0` | Phase 2 — insert grasped PCB along +Y into slot |

Robot runtime assets now live under `usd_model/usd_robot/`, and environment fixtures plus conversion sources live under `usd_model/usd_env/`.

**High-level episode flow (two-phase curriculum):**

1. **Phase 1 — Grasp** (`Isaac-WidowX-PCB-Grasp-v0`): PCB on rails, gripper open. Rewards shape approach, edge-centre alignment, and closure. Episode ends on successful edge grasp or timeout.
2. **Phase 2 — Insert** (`Isaac-WidowX-PCB-Insert-v0`): Reset snaps the PCB to closed jaws (kinematic grasp hold). Rewards shape +Y motion and slot insertion depth.

Recommended training order: **Grasp → Insert**.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers grasp/insert envs and applies the rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | WidowX scene, actions, observations, rewards, events, terminations, magazine/rail geometry. |
| `usd_model/env_v3/` | URDF + meshes + `convert_to_usd.py` → `usd_env/pcb_insertion_env.usd` (short axle rods / width cross-rods omitted) |
| `usd_model/usd_robot/` | Robot USD bundles (wxai follower). |
| `mdp_custom.py` | Custom MDP terms (regularization, rail reset, drop detection). |
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

Commands below assume you run training from the **Isaac Lab repository root** (the directory that contains `scripts/`).

## Train (rl-games PPO)

**Phase 1 — grasp only:**

```bash
conda activate isaac-sim   # or your env name
cd /path/to/IsaacLab   # repository root (contains scripts/)
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-Grasp-v0 --headless --num_envs 4096
```

**Phase 2 — insert only** (after grasp policy is reasonable, or from scratch with snapped grasp reset):

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-Insert-v0 --headless --num_envs 4096
```


## Evaluate / play a trained policy

Checkpoints are saved under phase-specific log folders (see TensorBoard section below).

### Grasp policy

```bash
cd /path/to/IsaacLab
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-Grasp-v0 \
    --num_envs 16
```

### Insert policy

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-Insert-v0 \
    --num_envs 16
```

The script finds the best saved model (`WidowX_PCB_Grasp_RL.pth` or `WidowX_PCB_Insert_RL.pth`) from the matching log directory.
Use `--use_last_checkpoint` to load the most-recent epoch checkpoint instead.
`--video` enables `--real-time` automatically (use `--no-real-time` to disable). Videos are saved at 30 fps so playback matches on-screen speed.

### Load a specific checkpoint

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-Grasp-v0 \
    --num_envs 16 \
    --checkpoint logs/rl_games/WidowX_PCB_Grasp_RL/widowx_pcb_grasp/nn/WidowX_PCB_Grasp_RL.pth
```

### Record a video of the rollout

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-Grasp-v0 \
    --num_envs 4 \
    --video \
    --video_length 500
```

Videos are saved under `logs/rl_games/<experiment>/videos/play/`.
`--video` turns on real-time stepping and records at **30 fps** (readable wall-clock speed).
Add `--headless` to render off-screen without the Isaac Sim GUI window.

To fix an already-recorded fast video (sim-tagged 125 fps):

```bash
ffmpeg -y -i input.mp4 -vf "setpts=PTS*(125/30)" -r 30 -c:v libx264 -crf 18 -pix_fmt yuv420p output_realtime.mp4
```

---

## TensorBoard

Training with rl-games **automatically** writes TensorBoard event files. No extra flags are required — `use_diagnostics: True` is already set in [`agents/rl_games_ppo_cfg.py`](agents/rl_games_ppo_cfg.py).

Run training from the **Isaac Lab repository root** (the directory that contains `scripts/`). If you start training from the `widowx_pcb` package folder, logs may land under an unexpected cwd.

### Where logs are saved

Each run folder contains `summaries/` (TensorBoard scalars), `nn/` (checkpoints), and `params/` (saved YAML configs).

| Task | Task ID | Log folder (under Isaac Lab root) |
|------|---------|-----------------------------------|
| Grasp | `Isaac-WidowX-PCB-Grasp-v0` | `logs/rl_games/WidowX_PCB_Grasp_RL/widowx_pcb_grasp/` |
| Insert | `Isaac-WidowX-PCB-Insert-v0` | `logs/rl_games/WidowX_PCB_Insert_RL/widowx_pcb_insert/` |

TensorBoard event files live in each run's `summaries/` subdirectory, e.g.:

`logs/rl_games/WidowX_PCB_Grasp_RL/widowx_pcb_grasp/summaries/events.out.tfevents.*`

### View logs

**Option A — project helper** (works from any cwd; resolves the Isaac Lab `logs/` folder automatically):

```bash
python /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/widowx_pcb/agents/monitor_tensorboard.py \
    --logdir /path/to/IsaacLab/logs --port 6006
```

**Option B — Isaac Lab wrapper** (from Isaac Lab root):

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir=logs --port=6006
```

Open <http://127.0.0.1:6006>. In the run selector, pick `WidowX_PCB_Grasp_RL` or `WidowX_PCB_Insert_RL`.

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
