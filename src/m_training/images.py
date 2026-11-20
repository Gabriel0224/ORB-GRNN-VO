# SPDX-License-Identifier: GPL-3.0-or-later
"""User images -> P1 observations -> normalized point features; labels are separate."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import yaml
from .frontend import DEFAULTS, build, export
from .geometry import relative_targets
from .runtime import atomic_json, atomic_npz, sha256


def resolve(base, value):
    return (Path(base) / Path(value).expanduser()).resolve()


def load_dataset(path, require_labels=False):
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or document.get("schema") != "m-images-v1" or not document.get("sequences"):
        raise ValueError("Expected an m-images-v1 dataset description")
    if set(document) - {"schema", "frontend", "sequences"}:
        raise ValueError("Unknown dataset fields")
    settings = {**DEFAULTS, **document.get("frontend", {})}
    records, used = {}, set()
    for name, raw in document["sequences"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Use safe alphanumeric sequence identifiers")
        if raw.get("split") not in ("train", "validation"):
            raise ValueError("Declare each sequence as train or validation")
        if ("images" in raw) == ("image_list" in raw):
            raise ValueError("Specify exactly one of images directory or image_list")
        if "images" in raw:
            folder = resolve(path.parent, raw["images"])
            if not folder.is_dir():
                raise FileNotFoundError(f"Image directory missing: {name}")
            def natural(p):
                return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", p.name)]
            files = sorted((p.resolve() for p in folder.glob(raw.get("pattern", "*"))
                            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}), key=natural)
        else:
            listing = resolve(path.parent, raw["image_list"])
            files = [resolve(listing.parent, line.strip()) for line in listing.read_text().splitlines() if line.strip()]
        if len(files) < 2 or len(set(files)) != len(files) or any(not p.is_file() for p in files):
            raise ValueError(f"Sequence {name} needs at least two existing, distinct images")
        if used.intersection(files):
            raise ValueError("Sequences may not share image files; keep training and validation separate")
        used.update(files)
        K = np.asarray(raw["intrinsics"], dtype=np.float64)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.array_equal(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0):
            raise ValueError("Supply calibrated zero-skew pinhole intrinsics for the input pixels")
        if raw.get("rectified") is not True:
            raise ValueError("Images must be undistorted/rectified beforehand; set rectified: true only when verified")
        item = {"name": name, "split": raw["split"], "images": files, "intrinsics": K, "settings": settings}
        if require_labels:
            if "poses" not in raw or raw.get("pose_convention") != "camera_to_world" or raw.get("translation_unit") != "meter":
                raise ValueError("Supervised training needs camera_to_world poses in meters; images alone are insufficient")
            item["poses"] = resolve(path.parent, raw["poses"])
            item["source_label_hashes"] = {"poses": sha256(item["poses"])}
            if "timestamps" in raw:
                timing = resolve(path.parent, raw["timestamps"])
                times = np.loadtxt(timing, dtype=np.float64).reshape(-1)
                item["source_label_hashes"]["timestamps"] = sha256(timing)
            else:
                fps = float(raw.get("fps", 0))
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError("Supply timestamps or a positive fps for image timing")
                times = np.arange(len(files), dtype=np.float64) / fps
            if len(times) != len(files) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("Timestamps must be finite, ordered and match the image count")
            item["times"] = times
            for key in ["valid_pairs", "source_ids"]:
                item[key] = resolve(path.parent, raw[key]) if key in raw else None
                if item[key] is not None:
                    item["source_label_hashes"][key] = sha256(item[key])
        records[name] = item
    return records


def candidates_from_observations(path, K):
    """No pose/label argument; preserve every selected geometric inlier."""
    arrays = {key: [] for key in ["x", "uv", "features", "previous"]}
    offsets, last_keys, last_features, last_uv = [0], None, np.empty(0, np.int64), None
    frame_count = 0
    with Path(path).open() as stream:
        for frame, line in enumerate(stream):
            record = json.loads(line)
            if record["input_frame"] != frame:
                raise ValueError("Native image order changed")
            keys = np.asarray(record["keypoints"], np.float64).reshape(-1, 4)
            matches = np.asarray(record["candidate_matches"], np.float64).reshape(-1, 7)
            if len(keys) != record["feature_count"] or not np.isfinite(keys).all() or not np.isfinite(matches).all():
                raise ValueError("Invalid native keypoints or candidates")
            geometry = record["geometry"]
            bits = geometry["inlier_mask"]
            if len(bits) != len(matches) or set(bits) - {"0", "1"}:
                raise ValueError("Invalid H/F inlier mask")
            mask = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) == ord("1")
            scores = np.asarray([geometry["score_h"], geometry["score_f"]], np.float32)
            model = geometry["model"]
            if not np.isfinite(scores).all() or np.any(scores < 0) or model not in ("none", "H", "F"):
                raise ValueError("Invalid H/F model")
            if model == "none":
                if mask.any():
                    raise ValueError("Rejected model cannot provide inliers")
            elif mask.sum() < 8 or scores.sum() <= 0 or model != ("H" if scores[0] / scores.sum() > .50 else "F"):
                raise ValueError("H/F selection does not follow the pinned P1 rule")
            if frame == 0:
                if len(matches):
                    raise ValueError("First image must not have an adjacent match list")
                last_keys = keys
                frame_count += 1
                continue
            if len(matches):
                if np.any(matches[:, :3] != np.rint(matches[:, :3])) or np.any(matches[:, :3] < 0):
                    raise ValueError("Noninteger native feature IDs or distances")
                a, b = matches[:, 0].astype(int), matches[:, 1].astype(int)
                if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b) or np.any(a >= len(last_keys)) or np.any(b >= len(keys)):
                    raise ValueError("Native matches must be one-to-one within adjacent images")
                np.testing.assert_array_equal(last_keys[a, :2], matches[:, 3:5])
                np.testing.assert_array_equal(keys[b, :2], matches[:, 5:7])
            selected = matches[mask]
            selected = selected[np.lexsort((selected[:, 1], selected[:, 0], selected[:, 2]))]
            features = selected[:, :2].astype(np.int64)
            uv = np.stack([last_keys[features[:, 0], :2], keys[features[:, 1], :2]], axis=1)
            previous = np.full(len(features), -1, np.int64)
            if frame > 1:
                _, old, current = np.intersect1d(last_features, features[:, 0], assume_unique=True, return_indices=True)
                previous[current] = old
                np.testing.assert_array_equal(last_uv[old, 1], uv[current, 0])
            xy = (uv - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]]
            arrays["x"].append(np.c_[xy[:, 0], xy[:, 1], xy[:, 1] - xy[:, 0]].astype(np.float32))
            for key, value in [("uv", uv), ("features", features), ("previous", previous)]:
                arrays[key].append(value)
            offsets.append(offsets[-1] + len(features))
            last_keys, last_features, last_uv = keys, features[:, 1], uv
            frame_count += 1
    if frame_count < 2:
        raise ValueError("Incomplete native observations")
    result = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    result["offsets"] = np.asarray(offsets, np.int64)
    return result


def load_poses(path, frames):
    poses = np.loadtxt(path, dtype=np.float64)
    if poses.shape != (frames, 12) or not np.isfinite(poses).all():
        raise ValueError("Pose text must have one finite 3x4 camera-to-world transform per image")
    result = np.repeat(np.eye(4)[None], frames, axis=0)
    result[:, :3] = poses.reshape(frames, 3, 4)
    R = result[:, :3, :3]
    if not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), rtol=0, atol=1e-5) or not np.allclose(np.linalg.det(R), 1, rtol=0, atol=1e-5):
        raise ValueError("Pose rotations must be proper SO(3) matrices")
    return result


def extract_dataset(dataset, output, binary, seed=42, with_labels=False):
    """Generate runtime intermediates from user images; never load bundled caches."""
    import cv2
    records = load_dataset(dataset, require_labels=with_labels)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new extraction output directory")
    output.mkdir(parents=True)
    manifest = {"schema": "m-image-candidates/v1", "uses_ground_truth_for_matching": False, "sequences": {}}
    reference = {"schema": "provided-camera-poses/v1", "sequences": {}}
    for name, item in records.items():
        files, K = item["images"], item["intrinsics"]
        image = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode the first image of {name}")
        height, width = image.shape
        image_hashes = [sha256(p) for p in files]
        observation = export(files, item["settings"], width, height, output / "native" / name, binary, seed)
        if [sha256(p) for p in files] != image_hashes:
            raise ValueError("Source images changed during P1 extraction")
        arrays = candidates_from_observations(observation, K)
        pairs = len(files) - 1
        if len(arrays["offsets"]) != pairs + 1:
            raise ValueError("Native export did not cover every real frame pair")
        cache = output / f"seq{name}_candidates.npz"
        atomic_npz(cache, **arrays)
        spec = {"sequence": name, "split": item["split"], "pairs": pairs, "all_geometric_matches": len(arrays["x"]),
                "P0_intrinsics": K.tolist(), "image_size": [width, height], "point_cache_sha256": sha256(cache),
                "image_sha256_in_order": image_hashes, "frontend_settings": item["settings"],
                "empty_pairs": int(np.count_nonzero(np.diff(arrays["offsets"]) == 0)), "seed": seed}
        if with_labels:
            if pairs < 10:
                raise ValueError("Training/validation needs at least 11 images per sequence")
            # Labels are read only after image-only matching and point features are complete.
            poses = load_poses(item["poses"], len(files))
            valid = np.ones(pairs, bool)
            sources = np.zeros(pairs, np.int64)
            if item["valid_pairs"]:
                raw = np.loadtxt(item["valid_pairs"], dtype=np.float64).reshape(-1)
                if raw.shape != (pairs,) or not np.isin(raw, [0, 1]).all():
                    raise ValueError("valid_pairs must contain one 0/1 value per frame pair")
                valid = raw.astype(bool)
            if item["source_ids"]:
                raw = np.loadtxt(item["source_ids"], dtype=np.float64).reshape(-1)
                if raw.shape != (pairs,) or not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
                    raise ValueError("source_ids must contain one integer per frame pair")
                sources = raw.astype(np.int64)
            v = np.lib.stride_tricks.sliding_window_view(valid, 10)
            ids = np.lib.stride_tricks.sliding_window_view(sources, 10)
            starts = np.flatnonzero(v.all(1) & (ids == ids[:, :1]).all(1))
            if item["split"] == "validation":
                if not valid.all() or np.any(sources != sources[0]):
                    raise ValueError("Complete validation requires one continuous, fully labeled sequence")
                starts = np.arange(pairs - 9)
            groups = sum((len(span) + 63) // 64 for span in np.split(starts, np.flatnonzero(np.diff(starts) != 1) + 1))
            if not len(starts):
                raise ValueError("Sequence contains no valid ten-pair training windows")
            ref = output / "reference"
            for sub in ["labels", "poses", "calib"]:
                (ref / sub).mkdir(parents=True, exist_ok=True)
            pose_path = ref / "poses" / f"{name}.txt"
            np.savetxt(pose_path, poses[:, :3].reshape(-1, 12), fmt="%.17g")
            label = ref / "labels" / f"seq{name}.npz"
            atomic_npz(label, targets=relative_targets(poses), valid_pair=valid, source_id=sources,
                       timestamps_s=item["times"], valid_start_10=starts)
            calibration = ref / "calib" / f"{name}.txt"
            projection = np.c_[K, np.zeros(3)].flatten()
            calibration.write_text("".join(f"P{i}: " + " ".join(format(x, ".17g") for x in projection) + "\n" for i in [0, 2]))
            spec.update(windows=len(starts), batches=groups, labels_sha256=sha256(label),
                        ground_truth_sha256=sha256(pose_path), calibration_sha256=sha256(calibration),
                        source_label_hashes=item["source_label_hashes"], label_source="user_provided_camera_poses")
            reference["sequences"][name] = {"labels_file": f"labels/seq{name}.npz", "labels_sha256": sha256(label),
                                           "cam0_pose_text_sha256": sha256(pose_path), "pose_frame": "camera"}
        manifest["sequences"][name] = spec
        print(f"P1 {name}: {len(files)} images, {pairs} pairs, {len(arrays['x'])} geometric inliers", flush=True)
    if with_labels:
        atomic_json(output / "reference" / "manifest.json", reference)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Extract P1 features from user images; images and calibration are required")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--with-labels", action="store_true", help="Also prepare supplied pose labels for supervised training")
    args = parser.parse_args()
    binary = build(args.build_dir, args.jobs)
    extract_dataset(args.dataset, args.output, binary, args.seed, args.with_labels)


if __name__ == "__main__":
    main()
