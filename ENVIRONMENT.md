# Development environment

The project is developed as a source checkout on top of Isaac Lab. The following is the recorded environment on 2026-09-30; use these versions as the compatibility baseline when setting up a machine. Simulator training has not been rerun solely to validate this record.

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS, x86-64 |
| Python | 3.11.15 (Conda environment `isaaclab`) |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | checkout [`d94504bcf91cb7ab7ff956a2d48ecd1bca82797a`](https://github.com/isaac-sim/IsaacLab/commit/d94504bcf91cb7ab7ff956a2d48ecd1bca82797a); installed `isaaclab` distribution reports 0.54.3 |
| PyTorch | 2.7.0+cu128 (CUDA 12.8 build) |
| Torchvision | 0.22.0+cu128 |
| RSL-RL | `rsl-rl-lib` 5.0.1 |
| Gymnasium | 1.2.1 |
| NumPy | 1.26.0 |
| NVIDIA driver | 580.126.09 (`nvidia-smi` reports CUDA compatibility up to 13.0) |

On Ubuntu 22.04 x86-64, the following setup follows the [Isaac Lab pip installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) while fixing the versions used here:

```bash
conda create -n isaaclab python=3.11.15
conda activate isaaclab
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout d94504bcf91cb7ab7ff956a2d48ecd1bca82797a
./isaaclab.sh --install rsl_rl
export ISAACLAB_ROOT="$PWD"
```

The upstream guide lists host dependencies and driver requirements. The commands above are a version-pinned installation recipe based on that guide; this repository has not rerun the full installation in a fresh environment. Check the installed versions before training. The project entry-point scripts add this repository's `source/` directory to `sys.path`; no wheel or editable installation of this repository is required.

The GitHub Actions job checks Python syntax and small CPU-only tests on Ubuntu 22.04 / Python 3.11. It does not install or launch Isaac Sim and is not a simulator compatibility test.
