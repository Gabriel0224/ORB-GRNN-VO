# SPDX-License-Identifier: GPL-3.0-or-later
"""Build and run the independent, pose-free P1 image frontend."""
import argparse
import json
from pathlib import Path
import subprocess
from .runtime import atomic_json, sha256


NATIVE = Path(__file__).parent / "native" / "p1"
DEFAULTS = {"nfeatures": 2000, "scale_factor": 1.2, "nlevels": 8, "fast_threshold": 20,
            "min_fast_threshold": 7, "search_radius": 100.0, "ratio_threshold": 0.9,
            "max_hamming": 50, "octave_tolerance": 1}


def source_hashes():
    return {p.relative_to(NATIVE).as_posix(): sha256(p) for p in sorted(NATIVE.rglob("*")) if p.is_file()}


def build(build_directory, jobs=2):
    build_directory = Path(build_directory).resolve()
    if build_directory == NATIVE.resolve() or build_directory.is_relative_to(NATIVE.resolve()):
        raise ValueError("Build outside the distributed native source directory")
    if jobs < 1:
        raise ValueError("Build jobs must be positive")
    manifest = json.loads((NATIVE / "UPSTREAM.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(NATIVE / name) != digest:
            raise ValueError(f"Pinned ORB extractor source changed: {name}")
    build_directory.mkdir(parents=True, exist_ok=True)
    with (build_directory / "build.log").open("a", encoding="utf-8") as log:
        try:
            subprocess.run(["cmake", "-S", str(NATIVE), "-B", str(build_directory), "-DCMAKE_BUILD_TYPE=Release"],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(["cmake", "--build", str(build_directory), "--parallel", str(jobs)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"P1 build failed. Install CMake, a C++ compiler, OpenCV development files and Eigen3; inspect {build_directory / 'build.log'}") from error
    binary = build_directory / "p1_export"
    if not binary.exists():
        raise FileNotFoundError("Expected p1_export from the CMake build")
    info = verify_binary(binary)
    atomic_json(build_directory / "build_manifest.json", {"sources": source_hashes(), "binary_sha256": sha256(binary), "native": info})
    return binary


def verify_binary(binary):
    info = json.loads(subprocess.check_output([str(binary), "--version"], text=True))
    expected = {"adapter_sha256": "export.cpp", "matcher_sha256": "matching.hpp", "geometry_sha256": "geometry.cpp",
                "geometry_header_sha256": "geometry.hpp", "build_sha256": "CMakeLists.txt",
                "extractor_sha256": "vendor/ORBextractor.cc", "header_sha256": "vendor/ORBextractor.h"}
    if any(info.get(key) != sha256(NATIVE / name) for key, name in expected.items()):
        raise ValueError("P1 executable does not match the distributed sources; rebuild it")
    if info.get("pose_estimation") is not False or info.get("schema_version") != 2:
        raise ValueError("Expected the pose-free P1 frontend")
    return info


def validate_settings(settings, width, height):
    if set(settings) != set(DEFAULTS):
        raise ValueError("Unsupported P1 frontend settings")
    import math
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in settings.values()):
        raise ValueError("P1 settings must be finite numbers")
    ints = ["nfeatures", "nlevels", "fast_threshold", "min_fast_threshold", "max_hamming", "octave_tolerance"]
    if any(type(settings[k]) is not int for k in ints):
        raise ValueError("Integer P1 settings cannot be fractional")
    if (min(width, height, settings["nfeatures"], settings["nlevels"], settings["min_fast_threshold"]) <= 0
            or settings["fast_threshold"] < settings["min_fast_threshold"] or settings["scale_factor"] <= 1
            or settings["search_radius"] <= 0 or not 0 < settings["ratio_threshold"] < 1
            or not 0 <= settings["max_hamming"] <= 256 or not 0 <= settings["octave_tolerance"] < settings["nlevels"]):
        raise ValueError("Invalid P1 extraction/matching settings")
    if min(width, height) / settings["scale_factor"] ** (settings["nlevels"] - 1) < 64:
        raise ValueError("Image pyramid becomes too small; supply larger images or explicitly reduce nlevels")


def export(images, settings, width, height, directory, binary, seed):
    """Only image paths and frontend settings cross the native process boundary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    validate_settings(settings, width, height)
    native = verify_binary(binary)
    if len(images) < 2 or any("\n" in str(p) or "\r" in str(p) for p in images):
        raise ValueError("At least two images and newline-free image paths are required")
    (directory / "images.txt").write_text("".join(str(Path(p).resolve()) + "\n" for p in images), encoding="utf-8")
    (directory / "settings.yaml").write_text("%YAML:1.0\n\n" + "\n".join(f"{k}: {v}" for k, v in {"width": width, "height": height, **settings}.items()) + "\n")
    observations = directory / "observations.jsonl"
    command = [str(binary), str(directory / "settings.yaml"), str(directory / "images.txt"), str(observations), str(seed)]
    with (directory / "frontend.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(directory / "frontend_manifest.json", {"native": native, "binary_sha256": sha256(binary),
                "images": len(images), "seed": seed, "settings": settings, "image_size": [width, height],
                "uses_ground_truth": False, "uses_camera_poses": False, "uses_future_images": False,
                "uses_loop_closure": False, "observation_sha256": sha256(observations)})
    return observations


def main():
    parser = argparse.ArgumentParser(description="Build P1 from the bundled source; no model weights or vocabulary required")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    print(build(args.build_dir, args.jobs))


if __name__ == "__main__":
    main()
