# <img src="assets/logo.png" alt="ivagrasp-isaaclab logo" width="48"> ivagrasp-isaaclab

English | [简体中文](README.zh-CN.md)

[![CI](https://github.com/HomeworldL/ivagrasp-isaaclab/actions/workflows/ci.yml/badge.svg)](https://github.com/HomeworldL/ivagrasp-isaaclab/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](ENVIRONMENT.md)
[![Code license: MIT](https://img.shields.io/badge/Code-MIT-2ea44f)](LICENSE)

Dexterous grasping of free-floating objects in [Isaac Lab](https://github.com/isaac-sim/IsaacLab). This research code trains a state-based teacher and distills a partial-observation student across configurable object sets. It includes stationary and moving-object tasks for LiberHand, Allegro, and Inspire hands. In multi-object training, different environments can use different objects from the selected set.

**Teacher (left) · Student (right), 16 parallel environments**

[![Synchronized teacher and student grasping varied objects in 16 environments](assets/media/multi-object-comparison.gif)](assets/media/multi-object-comparison.mp4)

Both views start before contact, show one grasp without an environment reset, and play at 0.4× the recorded speed. Each environment contains one object from the set. Open the [teacher](assets/media/multi-object-teacher.mp4) or [student](assets/media/multi-object-student.mp4) clip separately; closer views: [can teacher](assets/media/can-teacher.mp4) / [student](assets/media/can-student.mp4), [drill teacher](assets/media/drill-teacher.mp4) / [student](assets/media/drill-student.mp4).

The curve below shows mean episodic reward across four object-set training stages in an earlier run. It illustrates training progress, not grasp success rate.

![Mean episodic reward during four training stages](assets/media/training-reward.png)

## Install

Use Ubuntu 22.04, an NVIDIA GPU, and the `isaaclab` Conda environment. The commands below target the [recorded development stack](ENVIRONMENT.md): Python 3.11, Isaac Sim 5.1.0, PyTorch 2.7.0 with CUDA 12.8, and a specific Isaac Lab commit. Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) for host dependencies and driver requirements.

```bash
conda create -n isaaclab python=3.11.15 -y
conda activate isaaclab
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout d94504bcf91cb7ab7ff956a2d48ecd1bca82797a
(cd IsaacLab && ./isaaclab.sh --install rsl_rl)
export ISAACLAB_ROOT="$(pwd)/IsaacLab"

git clone https://github.com/HomeworldL/ivagrasp-isaaclab.git
cd ivagrasp-isaaclab
```

Run commands from the repository root. `./isaaclab.sh` delegates to `$ISAACLAB_ROOT/isaaclab.sh`; this repository is used directly from its source tree and does not need wheel installation. The train and play scripts also carry a legacy `/home/ccs/github/IsaacLab/scripts/reinforcement_learning/rsl_rl` Python search path. It can be absent on your machine; Isaac Lab and RSL-RL should be installed in the active environment.

## Prepare objects

This repository does not include object datasets or checkpoints. Follow the public [dexgrasp_sample guide](https://github.com/HomeworldL/dexgrasp_sample/blob/main/README.md#dataset-construction) to prepare object assets, convert them to USD, and build shape clusters and RL split metadata. Point the commands below at the generated directories:

```bash
export DATASET_ROOT=/path/to/dexgrasp_sample/datasets/objdata_YCB
export RL_SPLIT_DIR="$DATASET_ROOT/_meta/rl_split/<split_tag>"
```

The Python defaults still contain paths from the original workstation. Pass `--dataset-root` and `--rl-split-dir` explicitly; exporting the variables alone does not change those defaults. Choose an `--object-set-id` present in your split metadata.

## Train and play

Train a moving-object teacher on an object set:

```bash
./isaaclab.sh -p scripts/train_dexgrasp_moving.py \
  --headless --hand liberhand_right --mode teacher \
  --num-envs 128 --max-iterations 1000 \
  --dataset-root "$DATASET_ROOT" --rl-split-dir "$RL_SPLIT_DIR" \
  --object-set-id 'train@cluster@cluster0_topk1@s120_120'
```

Play a trained checkpoint on the same set:

```bash
./isaaclab.sh -p scripts/play_dexgrasp_moving.py \
  --hand liberhand_right --mode teacher \
  --checkpoint /path/to/model.pt \
  --dataset-root "$DATASET_ROOT" --rl-split-dir "$RL_SPLIT_DIR" \
  --object-set-id 'train@cluster@cluster0_topk1@s120_120'
```

For the stationary task, use `scripts/train_dexgrasp_float.py` and `scripts/play_dexgrasp_float.py` with the same arguments. Student training uses `--mode student --teacher-checkpoint /path/to/teacher.pt`; student playback uses `--mode student`. Use `--hand allegro_right` or `--hand inspire_right` to select another hand. Training writes checkpoints under `logs/rsl_rl/`.

The stationary task resets the object with zero linear and angular velocity. In the moving task, the default reset samples each velocity component independently: `vx ∈ [0, 0.45] m/s`, `vy, vz ∈ [-0.05, 0.05] m/s`, and `ωx, ωy, ωz ∈ [-0.35, 0.35] rad/s`. These are component ranges, not a single speed-magnitude range. The optional speed curriculum is disabled by default; set `ENABLE_SPEED_CURRICULUM=1` for training if you intend to use it.

For a checkpoint-free hand-asset viewer, run `./isaaclab.sh -p scripts/view_hand_usd.py --hands liberhand_right`. This requires Isaac Sim but no object dataset.

## License

Project code is [MIT-licensed](LICENSE). Hand assets may have different terms; see [third-party notices](THIRD_PARTY_NOTICES.md), particularly the Inspire Hand noncommercial license. Contributions are described in [CONTRIBUTING.md](CONTRIBUTING.md).
