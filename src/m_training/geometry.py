# SPDX-License-Identifier: GPL-3.0-or-later
"""Pose conventions and initial-pose trajectory alignment for validation."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def load_kitti_poses(path: str | Path) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    result = np.repeat(np.eye(4)[None], len(rows), axis=0)
    result[:, :3] = rows
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite KITTI poses")
    return result

def relative_targets(absolute: np.ndarray) -> np.ndarray:
    """Camera-to-world KITTI poses; delta = inv(T_t) @ T_(t+1)."""
    relative = np.linalg.solve(absolute[:-1], absolute[1:])
    # Extrinsic xyz = Rz(yaw) @ Ry(pitch) @ Rx(roll); output rotation first.
    roll_pitch_yaw = Rotation.from_matrix(relative[:, :3, :3]).as_euler("xyz")
    return np.concatenate((roll_pitch_yaw[:, ::-1], relative[:, :3, 3]), axis=1).astype(np.float32)

def vector_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[:3][::-1]).as_matrix()
    result[:3, 3] = pose[3:]
    return result

def accumulate_poses(relative_vectors: np.ndarray) -> np.ndarray:
    result = [np.eye(4)]
    for vector in relative_vectors:
        result.append(result[-1] @ vector_to_transform(vector))
    return np.stack(result)

def camera_from_cam0(calib_path: str | Path, camera: str) -> np.ndarray:
    """Rectified KITTI extrinsic T_camN_cam0 from the projection matrices.

    KITTI odometry pose files contain T_world_cam0 (see pykitti's odometry
    loader). The chosen image stream may be another rectified camera.
    """
    result = np.eye(4)
    if camera == "image_0":
        return result
    matrices = {}
    for line in Path(calib_path).read_text().splitlines():
        key, _, values = line.partition(":")
        if key in ("P0", "P" + camera.split("_")[-1]):
            matrices[key] = np.fromstring(values, sep=" ").reshape(3, 4)
    selected = matrices["P" + camera.split("_")[-1]]
    reference = matrices["P0"]
    result[:3, 3] = (np.linalg.solve(selected[:, :3], selected[:, 3]) -
                     np.linalg.solve(reference[:, :3], reference[:, 3]))
    return result

def trajectory_in_cam0(poses: np.ndarray, camera_extrinsic: np.ndarray) -> np.ndarray:
    """Convert a first-frame-normalized selected-camera trajectory for KITTI export."""
    return np.linalg.inv(camera_extrinsic) @ poses @ camera_extrinsic
def normalize_first_pose(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("Expected at least two finite 4x4 poses")
    return np.linalg.solve(poses[0], poses)
