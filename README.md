# ORB-GRNN-VO

**English** | [简体中文](README.zh-CN.md)

Monocular visual odometry combining an **ORB frontend** with a **GRNN-VO network**. This repository provides image preprocessing, supervised training, validation, and checkpoint recovery.

```text
Image sequences → ORB feature matching → Graph and recurrent modeling → Relative camera poses
```

## Installation

Use Python 3.10–3.12. GPU training requires a CUDA-compatible PyTorch installation.

```bash
sudo apt-get install build-essential cmake libopencv-dev libeigen3-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Training

Provide your own image sequences, camera intrinsics, and camera-pose labels. Use separate training and validation sequences with at least 11 images each.

Copy [the dataset example](configs/images.example.yaml) to `dataset.yaml` and update its paths and calibration. See the [input guide](docs/IMAGES.md) for the pose format.

```bash
grnn-vo-train --dataset ./dataset.yaml --config configs/grnn_vo.yaml \
  --work-dir ./runs/grnn_vo --device cuda:0
```

The command builds the ORB frontend, prepares features, and starts training. Add `--prepare-only` to prepare inputs first, or `--resume` to continue an existing run. Training settings are in [the configuration](configs/grnn_vo.yaml).

Checkpoints, logs, and validation trajectories are saved in the run directory. Generated caches are retained for reuse. This repository contains source code only; datasets, weights, and precomputed caches are not included.

## Attitude correction

Validation also applies the causal attitude-bias correction from Section 3.3: essential-matrix rotations from ORB inliers drive 176 scalar Kalman filters per rotation-vector component. Each output uses only earlier observations; translation is unchanged. Configure the candidate variances and initial variance (degrees²) in `attitude_correction`. Epoch trajectory files retain `prediction` and add `prediction_corrected`; validation metrics report `ate_first_pose_m` and `ate_first_pose_corrected_m` (first-pose alignment, fixed scale). Training loss, scheduling, and checkpoint selection still use the original uncorrected outputs.

## License

[GPL-3.0-or-later](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for ORB-SLAM3 and MLLA attribution.
