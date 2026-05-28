# PCB on-rail task — WidowX (Isaac Lab)

This package is an **Isaac Lab** manager-based RL task: a robot arm interacts with a **PCB** on a **magazine + conveyor** assembly (single aligned USD). Side guide-rail rods are omitted from the fixture USD. Training uses **rl-games** (PPO).

One simulation variant is registered:

| Robot | Task ID | Env config |
|--------|---------|------------|
| **WidowX** (Trossen `usd_model/usd_robot/wxai/wxai_follower.usd`) | `Isaac-WidowX-PCB-Grasp-v0` | Phase 1 — grasp trailing short-edge centre |
| | `Isaac-WidowX-PCB-Push-v0` | Phase 2 — push grasped PCB along +Y into slot |
| | `Isaac-WidowX-PCB-v0` | Both phases in one episode (gated rewards) |

Robot runtime assets now live under `usd_model/usd_robot/`, and environment fixtures plus conversion sources live under `usd_model/usd_env/`.

**High-level episode flow (two-phase curriculum):**

1. **Phase 1 — Grasp** (`Isaac-WidowX-PCB-Grasp-v0`): PCB on rails, gripper open. Rewards shape approach, edge-centre alignment, and closure. Episode ends on successful edge grasp or timeout.
2. **Phase 2 — Push** (`Isaac-WidowX-PCB-Push-v0`): Reset snaps the PCB to closed jaws (kinematic grasp hold). Rewards shape +Y motion and slot insertion depth.
3. **Full task** (`Isaac-WidowX-PCB-v0`): Same obs/actions; grasp rewards active until edge grasp, then push rewards only.

Recommended training order: **Grasp → Push → (optional) Full** fine-tune.

---

## Repository layout

| Path | Role |
|------|------|
| `__init__.py` | Registers `Isaac-WidowX-PCB-v0` and applies the rl-games log-std safety patch. |
| `widowx_pcb_env_cfg.py` | WidowX scene, actions, observations, rewards, events, terminations, magazine/rail geometry. |
| `usd_model/env_v3/` | URDF + meshes + `convert_to_usd.py` → `usd_env/pcb_insertion_env.usd` (short axle rods / width cross-rods omitted) |
| `usd_model/usd_robot/` | Robot USD bundles (wxai follower). |
| `mdp_custom.py` | Custom MDP terms (regularization, rail reset, drop detection). |
| `agents/` | PPO config (`WidowXPcbPPOCfg` in `rl_games_ppo_cfg.py`), log-std safety helper, TensorBoard monitor script. |

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

**Phase 2 — push only** (after grasp policy is reasonable, or from scratch with snapped grasp reset):

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-Push-v0 --headless --num_envs 4096
```

**Full two-phase episode** (single policy, gated rewards):

```bash
python scripts/reinforcement_learning/rl_games/train.py --task Isaac-WidowX-PCB-v0 --headless --num_envs 4096
```


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
`--video` now enables `--real-time` automatically (use `--no-real-time` to disable). Videos are saved at 30 fps so playback matches on-screen speed.

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
`--video` turns on real-time stepping and records at **30 fps** (readable wall-clock speed).
Add `--headless` to render off-screen without the Isaac Sim GUI window.

To fix an already-recorded fast video (sim-tagged 125 fps):

```bash
ffmpeg -y -i input.mp4 -vf "setpts=PTS*(125/30)" -r 30 -c:v libx264 -crf 18 -pix_fmt yuv420p output_realtime.mp4
```

---



From the Isaac Lab repo (adjust path if your checkout differs):

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/widowx_pcb/agents/monitor_tensorboard.py --logdir logs --port 6006
```

Open <http://127.0.0.1:6006> and watch policy/value loss, entropy, KL, and episodic return.

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

1. In `RewardsCfg`, balance insertion / push terms vs **`action_rate_penalty`**.
2. In PPO (`agents/rl_games_ppo_cfg.py`), raise **entropy** slightly or decay it more slowly if the policy collapses early.
3. Log **per-term rewards** in TensorBoard if available, to see which term is flat.
