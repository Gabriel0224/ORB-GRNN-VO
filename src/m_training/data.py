# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only ORB candidate inputs; independent causal windows and native grouping."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from .geometry import load_kitti_poses, relative_targets, camera_from_cam0, normalize_first_pose
from .point_mlla import spatial_neighbors
from .runtime import sha256


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest path must stay within the input directory")
    return path


class Sequence:
    def __init__(self, folder, sequence, config, neighbors=True):
        self.sequence = sequence
        if sequence not in config["data"]["train_sequences"] + [config["data"]["validation_sequence"]]:
            raise ValueError("The training package only reads training and validation sequences")
        folder = Path(folder).resolve()
        ref = folder / "reference"
        manifest = json.loads((folder / "manifest.json").read_text())
        info = json.loads((ref / "manifest.json").read_text())
        spec, rs = manifest["sequences"][sequence], info["sequences"][sequence]
        names = {"points": folder / f"seq{sequence}_candidates.npz",
                 "labels": contained(ref, rs["labels_file"]),
                 "poses": ref / "poses" / f"{sequence}.txt", "calib": ref / "calib" / f"{sequence}.txt"}
        expected = {"points": spec["point_cache_sha256"], "labels": spec["labels_sha256"],
                    "poses": spec["ground_truth_sha256"], "calib": spec["calibration_sha256"]}
        if expected["labels"] != rs["labels_sha256"] or expected["poses"] != rs["cam0_pose_text_sha256"]:
            raise ValueError("Input and reference manifests disagree")
        self.file_hashes = {}
        for key, path in names.items():
            actual = sha256(path)
            if actual != expected[key]:
                raise ValueError(f"Sequence {sequence}: {key} checksum mismatch")
            self.file_hashes[path.relative_to(folder).as_posix()] = actual
        for path in [folder / "manifest.json", ref / "manifest.json"]:
            self.file_hashes[path.relative_to(folder).as_posix()] = sha256(path)
        with np.load(names["points"], allow_pickle=False) as z:
            offsets, x, uv, features, previous = [z[k].copy() for k in ("offsets", "x", "uv", "features", "previous")]
        pairs = int(spec["pairs"])
        if (x.dtype != np.float32 or uv.dtype != np.float64 or x.shape != (len(uv), 6)
                or uv.shape != (len(x), 2, 2) or features.shape != (len(x), 2)
                or previous.shape != (len(x),) or len(offsets) != pairs + 1
                or offsets[0] != 0 or offsets[-1] != len(x) or np.any(np.diff(offsets) < 0)
                or len(x) != spec["all_geometric_matches"]):
            raise ValueError(f"Sequence {sequence}: unexpected ORB candidate layout")
        if not all(np.isfinite(v).all() for v in [x, uv]) or np.any(previous < -1):
            raise ValueError("Invalid candidate values")
        self.K = np.asarray(spec["P0_intrinsics"], dtype=np.float64)
        projection = {}
        for line in names["calib"].read_text().splitlines():
            key, _, values = line.partition(":")
            if key in ("P0", "P2"):
                projection[key] = np.fromstring(values, sep=" ").reshape(3, 4)
        np.testing.assert_array_equal(self.K, projection["P0"][:, :3])
        np.testing.assert_array_equal(self.K, projection["P2"][:, :3])
        xy = (uv - [self.K[0, 2], self.K[1, 2]]) / [self.K[0, 0], self.K[1, 1]]
        np.testing.assert_array_equal(x, np.c_[xy[:, 0], xy[:, 1], xy[:, 1] - xy[:, 0]].astype(np.float32))
        self.frames = [x[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        self.uv = [uv[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        self.feature_pairs = [features[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        self.previous = [previous[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        for t, (f, p) in enumerate(zip(self.feature_pairs, self.previous)):
            if len(np.unique(f[:, 0])) != len(f) or len(np.unique(f[:, 1])) != len(f):
                raise ValueError("A feature participates in more than one match")
            valid = p >= 0
            if t == 0 and valid.any():
                raise ValueError("First pair cannot have temporal parents")
            if t:
                if np.any(p[valid] >= len(self.frames[t - 1])):
                    raise ValueError("Temporal parent outside the previous frame pair")
                np.testing.assert_array_equal(self.feature_pairs[t - 1][p[valid], 1], f[valid, 0])
                np.testing.assert_array_equal(self.uv[t - 1][p[valid], 1], self.uv[t][valid, 0])
        length = config["data"]["window_length"]
        with np.load(names["labels"], allow_pickle=False) as z:
            self.targets = z["targets"].copy()
            self.valid_pair = z["valid_pair"].copy()
            self.source_id = z["source_id"].copy()
            self.times = z["timestamps_s"].copy()
            if sequence in config["data"]["train_sequences"]:
                v = np.lib.stride_tricks.sliding_window_view(self.valid_pair, length)
                s = np.lib.stride_tricks.sliding_window_view(self.source_id, length)
                self.starts = np.flatnonzero(v.all(1) & (s == s[:, :1]).all(1))
                np.testing.assert_array_equal(self.starts, z["valid_start_10"])
            else:
                self.starts = np.arange(pairs - length + 1)
        if self.targets.dtype != np.float32 or self.targets.shape != (pairs, 6) or not np.isfinite(self.targets).all():
            raise ValueError("Expected finite FP32 relative six-axis targets")
        gt = load_kitti_poses(names["poses"])
        self.gt = normalize_first_pose(gt)
        pose_frame = rs.get("pose_frame", "cam0")
        if pose_frame not in ("cam0", "camera"):
            raise ValueError("Unsupported pose coordinate convention")
        if config["data"].get("protocol", "kitti_fused10") == "custom_images" and pose_frame != "camera":
            raise ValueError("Custom-image labels must use the input camera pose convention")
        self.extrinsic = camera_from_cam0(names["calib"], "image_2") if pose_frame == "cam0" else np.eye(4)
        if len(gt) != pairs + 1:
            raise ValueError("Pose count differs from frame-pair count")
        np.testing.assert_allclose(self.targets, relative_targets(gt @ np.linalg.inv(self.extrinsic)), atol=1e-7, rtol=0)
        self.groups = []
        for span in np.split(self.starts, np.flatnonzero(np.diff(self.starts) != 1) + 1):
            for i in range(0, len(span), config["training"]["batch_size"]):
                self.groups.append(span[i:i + config["training"]["batch_size"]].tolist())
        if len(self.groups) != spec["batches"] or len(self.starts) != spec["windows"]:
            raise ValueError("Window counts differ from the frozen manifest")
        self.neighbors = ([spatial_neighbors(u[:, 1], f[:, 1]) for u, f in zip(self.uv, self.feature_pairs)]
                          if neighbors else None)


def pack(sequence, starts, length=10, targets=True):
    if sequence.neighbors is None:
        raise ValueError("Spatial neighbors have not been prepared")
    if not starts or length < 1 or any(s < 0 or s + length > len(sequence.frames) for s in starts):
        raise ValueError("Window outside the sequence")
    xs, counts, ys, positions, neighbors, parents, offsets = [], [], [], [], [], [], [0]
    for start in starts:
        previous_offset = None
        for t in range(start, start + length):
            frame = sequence.frames[t]
            n = len(frame)
            xs.append(frame)
            counts.append(n)
            positions.append((sequence.uv[t][:, 1, :] / 8).astype(np.float32))
            ni = sequence.neighbors[t]
            neighbors.append(np.where(ni >= 0, ni + offsets[-1], -1))
            prev = sequence.previous[t]
            parents.append(np.full(n, -1, np.int64) if t == start else np.where(prev >= 0, prev + previous_offset, -1))
            previous_offset = offsets[-1]
            offsets.append(offsets[-1] + n)
        if targets:
            ys.append(sequence.targets[start:start + length])
    return {"x": torch.from_numpy(np.concatenate(xs).astype(np.float32, copy=False)),
            "counts": torch.tensor(counts, dtype=torch.long), "counts_list": tuple(counts),
            "offsets_list": tuple(offsets), "offsets": torch.tensor(offsets, dtype=torch.long),
            "previous": torch.from_numpy(np.concatenate(parents).astype(np.int64, copy=False)),
            "positions": torch.from_numpy(np.concatenate(positions)),
            "neighbors": torch.from_numpy(np.concatenate(neighbors)),
            "target": torch.from_numpy(np.stack(ys)) if targets else None,
            "batch_size": len(starts), "length": length, "sequence": sequence.sequence, "starts": list(map(int, starts))}


class PreparedBatches:
    """Bounded CPU cache; no experiment-specific global paths or GPU allocations."""
    def __init__(self, root, records, implementation="reference", cache_size=0):
        self.root, self.records = Path(root), records
        self.implementation, self.cache_size, self.cache = implementation, cache_size, OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self.records[index]
        path = contained(self.root, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Prepared batch checksum mismatch: {record['path']}")
        batch = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if self.implementation == "compact":
            from .execution import prepare
            from .compact_graph import prepare as compact_prepare
            batch = compact_prepare(prepare(batch))
        if self.cache_size:
            self.cache[index] = batch
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return batch
