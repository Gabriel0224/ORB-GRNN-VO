# Input guide

**English** | [简体中文](IMAGES.zh-CN.md)

Start from [images.example.yaml](../configs/images.example.yaml). Paths are relative to the dataset YAML file. Replace the example intrinsics with your camera calibration.

## Images and calibration

- Supply one or more `train` sequences and one separate `validation` sequence, each with at least 11 images.
- Use `images` plus `pattern` for a directory in natural filename order, or `image_list` for an explicit ordered list. Paths inside an image list are relative to that list.
- Supply undistorted/rectified images, set `rectified: true`, and provide the matching zero-skew 3×3 `intrinsics`. Image size must remain constant within a sequence. No automatic resizing is applied.
- Keep training and validation data separate. The ORB frontend processes adjacent images without using pose labels or future frames.

## Pose labels

Supervised training requires one camera-to-world pose per image, in the same camera frame as the images. Set `pose_convention: camera_to_world` and `translation_unit: meter`.

The `poses` text file contains one row-major 3×4 matrix per line:

```text
r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
```

Rotations must be valid rotation matrices. Relative targets are computed as `inverse(T[t]) @ T[t+1]`. Provide either increasing `timestamps` in seconds or a positive `fps`.

Optional `valid_pairs` and `source_ids` files contain one value per adjacent image pair. Training windows must contain valid pairs from a single source. Validation requires a fully labeled continuous sequence.

## Generated files

`generated-inputs/` stores ORB observations, point features, and prepared labels; `prepared/` stores training batches. Both remain after training. The [intermediate format](INPUTS.md) is documented for integrations.

To extract features without pose labels:

```bash
grnn-vo-extract --dataset ./dataset.yaml --output ./generated/orb \
  --build-dir ./build/orb
```
