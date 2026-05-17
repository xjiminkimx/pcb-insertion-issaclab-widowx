# PCB on-rail task — WidowX (Isaac Lab)

This package is an **Isaac Lab** manager-based RL task: a robot arm interacts with a **PCB** on a **magazine + guide rail** assembly (single aligned USD). **Target-slot distance rewards and slot observations have been removed** so you can add forward-push / rail objectives separately. Training uses **rl-games** (PPO).

One simulation variant is registered:

| Robot | Task ID | Env config |
|--------|---------|------------|
| **WidowX** (Trossen `usd_model/usd_robot/wxai/wxai_follower.usd`) | `Isaac-WidowX-PCB-v0` | `widowx_pcb_env_cfg.py` → `WidowXPcbEnvCfg` |

Robot runtime assets now live under `usd_model/usd_robot/`, and environment fixtures plus conversion sources live under `usd_model/usd_env/`.

**High-level episode flow:**

1. Reset places the PCB on the rail pose (see `_PCB_INIT_*` and reset events).
2. Dense rewards are regularization-style (height, floor, tilt, etc.) — no slot target.
3. Terminations: PCB too low / dropped; success termination is left for you to define with the new task reward.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers `Isaac-WidowX-PCB-v0` and applies the rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | WidowX scene, actions, observations, rewards, events, terminations, magazine/rail geometry. |
| `usd_model/` | Unified asset root: `usd_env/` contains fixture USDs and conversion sources, and `usd_robot/` contains robot USD bundles. |
| `mdp_custom.py` | Custom MDP terms (regularization, rail reset, drop detection). |
| `agents/` | PPO config (`WidowXPcbPPOCfg` in `rl_games_ppo_cfg.py`), log-std safety helper, TensorBoard monitor script. |
| `jetcobot_assets/` | URDF/USD/meshes (legacy / alternate robot assets). |

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

**WidowX:**

```bash
conda activate isaac-sim   # or your env name
cd /path/to/IsaacLab   # repository root (contains scripts/)
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0 --headless --num_envs 4096
```

If Isaac Sim startup is unstable on newer GPUs or drivers, prefer **headless** mode with safer renderer flags:

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0 --headless --rendering_mode performance \
  --kit_args "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/rtx/raytracing/cached/enabled=false"
```

Variant:

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0 --video
```

---

## Evaluate / play a trained policy

Checkpoints are saved to `logs/rl_games/WidowX_PCB_RL/widowx_pcb/nn/` during training.

### Load the best checkpoint automatically

```bash
cd /path/to/IsaacLab
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-v0 \
    --num_envs 16
```

The script finds `WidowX_PCB_RL.pth` (the best saved model) automatically from the log directory.
Use `--use_last_checkpoint` to load the most-recent epoch checkpoint instead.
Use `--real-time` to throttle stepping to wall-clock speed.

### Load a specific checkpoint

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-v0 \
    --num_envs 16 \
    --checkpoint logs/rl_games/WidowX_PCB_RL/widowx_pcb/nn/last_WidowX_PCB_RL_ep_75_rew_0.51598245.pth
```

### Record a video of the rollout

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-WidowX-PCB-v0 \
    --num_envs 4 \
    --video \
    --video_length 500
```

Video is saved to `logs/rl_games/WidowX_PCB_RL/widowx_pcb/videos/play/`.
Add `--headless` to render off-screen without the Isaac Sim GUI window.

---



From the Isaac Lab repo (adjust path if your checkout differs):

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/widowx_pcb/agents/monitor_tensorboard.py --logdir logs --port 6006
```

Open <http://127.0.0.1:6006> and watch policy/value loss, entropy, KL, and episodic return.

---

## Important task knobs

### Fixture pose vs USD

Align `_MAG_POS` / `_MAG_ROT_WXYZ` in `widowx_pcb_env_cfg.py` with your imported **magazine + rail** asset in world frame. Analytic `_GUIDE_RAIL_*` may not match `usd_model/usd_env/magazine.usd` collision—if the **green PCB clips into white rails**, raise **`_PCB_SPAWN_Z_BIAS`** (and optionally `collision_props.contact_offset` / `rest_offset` on PCB + magazine) until the board sits on the rail tops in the contact view.

### Physics and collision

If the PCB intersects the fixture: tune CCD and contact offsets in the env config and collision meshes in USD.

### Reset pose and drop termination

Tune **`pcb_tilt_excessive`**, **`pcb_long_axis_not_horizontal`**, **`pcb_fallen_below_rail`**, **`pcb_dropped`**, and **`arm_idle`** (`_ARM_IDLE_MIN_STEPS`, `_ARM_IDLE_MAX_ABS_VEL_RAD_S`) if episodes reset too aggressively—or not enough when the PCB slips / the policy freezes.

### Grippers and observations

- **WidowX:** fingertip bodies `gripper_left` / `gripper_right`, drive joint `left_carriage_joint`.

### If mean reward plateaus (policy idles near the edge)

1. In `RewardsCfg`, balance insertion / push terms vs **`action_rate_penalty`**.
2. In PPO (`agents/rl_games_ppo_cfg.py`), raise **entropy** slightly or decay it more slowly if the policy collapses early.
3. Log **per-term rewards** in TensorBoard if available, to see which term is flat.

---

## Troubleshooting

- **GPU / RTX Blackwell (RTX 5080, 5090, 5060 Ti, …) — segfault at startup (`librtx.scenedb.plugin`, `libcarb.scenerenderer-rtx`, ~2–4 s):** This is usually **Kit + Vulkan + RTX** on a **driver or GPU generation** combination Isaac Sim was not validated against yet — not a broken CUDA install if `nvidia-smi` works.
  1. Train **headless** and use the direct `train.py` command with the `--kit_args` line in the Train section above.
  2. Install the **NVIDIA driver branch** listed for your **Isaac Sim version** in NVIDIA’s release notes / download page. Community reports often show **580.x** working when **595+** crashes on Blackwell; pick the validated branch before chasing application bugs.
  3. Avoid the **GUI** path while debugging (`./isaac-sim.sh` without headless); it pulls a heavier RTX path.
  4. If it still crashes, try **Isaac Sim / Isaac Lab updates** (patch releases often add Blackwell fixes) or ask on [NVIDIA Isaac Sim forums](https://forums.developer.nvidia.com/c/omniverse/simulation/69) with your **Kit log**, **driver version**, and **GPU model**.

- **Body name errors:** Match `SceneEntityCfg("robot", body_names=...)` to link names in your robot asset — **WidowX:** `gripper_left`, `gripper_right`.
- **Penetration at reset:** Slightly increase vertical `pos_offset`, soften edge offsets, or check gripper initial opening.

---

## Quick workflow

1. Match `_MAG_*` / `_PCB_INIT_*` to your combined USD (in the env cfg for the task you train).
2. Add rail-forward rewards and success criteria as needed.
3. Train `Isaac-WidowX-PCB-v0` and monitor TensorBoard.
4. Tune reset and termination thresholds if needed.
