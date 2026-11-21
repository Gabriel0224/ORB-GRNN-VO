# SPDX-License-Identifier: GPL-3.0-or-later
"""Causal rotation-bias correction using essential-matrix observations."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_SETTINGS = {
    "enabled": True,
    "q_min_deg2": 1e-7,
    "q_max_deg2": 1e-4,
    "q_count": 16,
    "eta_min_deg2": 1e-3,
    "eta_max_deg2": 1e-1,
    "eta_count": 11,
    "v0_deg2": 1e-3,
}


def filter_parameters(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("attitude_correction must be a mapping")
    settings = {} if settings is None else settings
    if set(settings) - DEFAULT_SETTINGS.keys():
        raise ValueError("Unknown attitude-correction setting")
    values = {**DEFAULT_SETTINGS, **settings}
    if type(values["enabled"]) is not bool:
        raise ValueError("attitude_correction.enabled must be boolean")
    for key in ("q_min_deg2", "q_max_deg2", "eta_min_deg2", "eta_max_deg2", "v0_deg2"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    grids = []
    for name in ("q", "eta"):
        low, high, count = (values[f"{name}_min_deg2"], values[f"{name}_max_deg2"], values[f"{name}_count"])
        if type(count) is not int or count < 1 or high < low:
            raise ValueError(f"Invalid {name} candidate grid")
        grids.append(np.geomspace(low, high, count))
    q, eta = np.meshgrid(*grids, indexing="ij")
    return q.ravel(), eta.ravel(), float(values["v0_deg2"])


def estimate_geometric_rotation(matches, intrinsics):
    """Return R.T in the pose-label convention, or None. Use f=(fx+fy)/2."""
    uv = np.asarray(matches, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0):
        raise ValueError("Expected finite zero-skew camera intrinsics")
    if uv.ndim != 3 or uv.shape[1:] != (2, 2):
        raise ValueError("Matches must have shape (N, 2, 2)")
    if len(uv) < 8 or not np.isfinite(uv).all():
        return None
    xy = (uv - K[:2, 2]) / [K[0, 0], K[1, 1]]
    first, second = np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
    camera = np.eye(3)
    try:
        essential, mask = cv2.findEssentialMat(first, second, camera, method=cv2.RANSAC,
                                              prob=0.999, threshold=2.0 / (K[0, 0] + K[1, 1]))
    except cv2.error:
        return None
    if essential is None or mask is None or np.asarray(mask).size != len(uv):
        return None
    essential = np.asarray(essential, dtype=np.float64)
    if essential.ndim != 2 or essential.shape[1] != 3 or essential.shape[0] % 3:
        return None
    mask = (np.asarray(mask).reshape(-1, 1) != 0).astype(np.uint8)
    best, best_count = None, 0
    for candidate in essential.reshape(-1, 3, 3):
        if not np.isfinite(candidate).all():
            continue
        try:
            count, rotation, translation, _ = cv2.recoverPose(candidate, first, second, camera, mask=mask.copy())
        except cv2.error:
            continue
        if (count > best_count and np.isfinite(rotation).all() and np.isfinite(translation).all()
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                and abs(np.linalg.det(rotation) - 1) <= 1e-6 and np.linalg.norm(translation) > 0):
            best, best_count = rotation.T.copy(), int(count)
    return best


def estimate_sequence_rotations(matches_by_pair, intrinsics):
    rotations = np.repeat(np.eye(3)[None], len(matches_by_pair), axis=0)
    valid = np.zeros(len(matches_by_pair), dtype=bool)
    for t, matches in enumerate(matches_by_pair):
        rotation = estimate_geometric_rotation(matches, intrinsics)
        if rotation is not None:
            rotations[t], valid[t] = rotation, True
    return rotations, valid


class RotationBiasFilter:
    """Three independent scalar-filter banks; rotation vectors are in degrees."""
    def __init__(self, settings=None):
        self.q, self.eta, v0 = filter_parameters(settings)
        self.bias = np.zeros((3, len(self.q)), dtype=np.float64)
        self.variance = np.full_like(self.bias, v0)
        self.log_likelihood = np.zeros_like(self.bias)

    def step(self, network_deg, geometry_deg=None):
        network = np.asarray(network_deg, dtype=np.float64)
        geometry = None if geometry_deg is None else np.asarray(geometry_deg, dtype=np.float64)
        if network.shape != (3,) or not np.isfinite(network).all():
            raise ValueError("Expected a finite three-component network rotation vector")
        if geometry is not None and (geometry.shape != (3,) or not np.isfinite(geometry).all()):
            raise ValueError("Expected a finite geometry rotation vector or None")
        weights = np.exp(self.log_likelihood - self.log_likelihood.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        corrected = network - np.sum(weights * self.bias, axis=1)
        # Current geometry updates the next output, never the current one.
        self.variance += self.q[None, :]
        if geometry is not None:
            innovation = (geometry - network)[:, None] + self.bias
            innovation_variance = self.variance + self.eta[None, :]
            self.log_likelihood -= 0.5 * (np.log(innovation_variance) + innovation ** 2 / innovation_variance)
            gain = self.variance / innovation_variance
            self.bias -= gain * innovation
            self.variance *= 1 - gain
        return corrected


def correct_relative_poses(relative, geometric_rotations, valid, settings=None):
    """Correct one sequence; preserve translation and reset filter state."""
    relative = np.asarray(relative, dtype=np.float64)
    rotations = np.asarray(geometric_rotations, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if relative.ndim != 2 or relative.shape[1] != 6 or not np.isfinite(relative).all():
        raise ValueError("Expected finite relative poses with shape (N, 6)")
    if rotations.shape != (len(relative), 3, 3) or valid.shape != (len(relative),):
        raise ValueError("Geometry and predictions must have identical frame-pair counts")
    bank = RotationBiasFilter(settings)
    corrected = relative.copy()
    correction_deg = np.zeros((len(relative), 3), dtype=np.float64)
    if not len(relative):
        return corrected, correction_deg
    observed = rotations[valid]
    if (not np.isfinite(observed).all()
            or not np.allclose(observed.transpose(0, 2, 1) @ observed, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(observed), 1, atol=1e-6, rtol=0)):
        raise ValueError("Valid geometry rotations must lie in SO(3)")
    network = Rotation.from_euler("xyz", relative[:, :3][:, ::-1]).as_rotvec(degrees=True)
    geometry = np.zeros_like(network)
    if valid.any():
        geometry[valid] = Rotation.from_matrix(observed).as_rotvec(degrees=True)
    filtered = np.empty_like(network)
    for t, vector in enumerate(network):
        filtered[t] = bank.step(vector, geometry[t] if valid[t] else None)
    correction_deg[:] = network - filtered
    changed = np.any(correction_deg != 0, axis=1)
    if changed.any():
        corrected[changed, :3] = Rotation.from_rotvec(filtered[changed], degrees=True).as_euler("xyz")[:, ::-1]
    return corrected, correction_deg
