# SPDX-License-Identifier: GPL-3.0-or-later
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {"seed", "data", "model", "training", "execution", "saving"}
    if not required.issubset(config):
        raise ValueError(f"Missing configuration sections: {required - config.keys()}")
    data = config["data"]
    protocol = data.get("protocol", "kitti_fused10")
    train, validation = data["train_sequences"], data["validation_sequence"]
    if (not isinstance(train, list) or not train or len(set(train)) != len(train)
            or validation in train or not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in train + [validation])):
        raise ValueError("Use distinct, safe training and validation sequence names")
    if protocol == "kitti_fused10":
        if train != ["00", "02", "07", "08"] or validation != "05":
            raise ValueError("The frozen KITTI protocol uses training 00/02/07/08 and validation 05")
    elif protocol != "custom_images":
        raise ValueError("Unknown data protocol")
    if config["data"]["window_length"] != 10 or config["training"]["batch_size"] != 64:
        raise ValueError("This GRNN-VO configuration requires ten-pair windows and batch 64")
    if data["window_stride"] != 1 or data["camera"] not in ("image_2", "camera"):
        raise ValueError("GRNN-VO uses stride-one windows and an explicit camera frame")
    expected_reference = "fused10" if protocol == "kitti_fused10" else "provided_poses"
    if data["input"] != "p1_all_float_inliers" or data["reference"] != expected_reference:
        raise ValueError("Input/reference definition disagrees with the data protocol")
    if protocol == "custom_images" and data["camera"] != "camera":
        raise ValueError("Custom-image poses must describe the calibrated input camera")
    if data["expected_counts"] is not None and (set(data["expected_counts"]) != {"train_windows", "train_batches", "validation_windows", "validation_batches"}
            or any(type(v) is not int or v < 1 for v in data["expected_counts"].values())):
        raise ValueError("Expected counts must be positive integers, or null for custom data")
    if protocol == "kitti_fused10" and data["expected_counts"] is None:
        raise ValueError("Frozen KITTI counts cannot be omitted")
    model_name = config["model"].get("name")
    if model_name not in ("GRNN-VO", "M"):
        raise ValueError("Unknown model name")
    if config["model"] != {"name": model_name, "parameters": 345542, "point_width": 64,
                            "graph_layers": 2, "lstm_input": 128, "lstm_hidden": 128,
                            "lstm_layers": 2, "dropout": 0.2, "geometry_input": False}:
        raise ValueError("Model definition differs from the published GRNN-VO architecture")
    t = config["training"]
    from .schedule import V3_LEARNING_RATES
    if tuple(t["learning_rates"]) != V3_LEARNING_RATES or t["patience"] != 20:
        raise ValueError("GRNN-VO uses the ten-stage validation schedule with patience 20")
    if not 1 <= t["max_epochs"] <= 300:
        raise ValueError("max_epochs must be between 1 and 300")
    if t["axis_weights"] != [300, 300, 300, 1, 1, 1]:
        raise ValueError("The GRNN-VO pose-loss weights must be 300/300/300/1/1/1")
    if t["weight_decay"] != 0 or t["gradient_clip"] != 10 or t["precision"] != "float32":
        raise ValueError("Expected weight_decay=0, clip=10 and FP32")
    e = config["execution"]
    if e["implementation"] not in ("reference", "compact") or e["frame_chunk_size"] != 32:
        raise ValueError("Use reference or compact execution with canonical chunks of 32")
    if e["cpu_threads"] < 1 or e["batch_cache_size"] < 0 or e["retain_nodes"] < 0:
        raise ValueError("Invalid execution resource setting")
    if config["saving"]["last_every_steps"] < 1 or config["saving"]["selection_last_epochs"] != 50:
        raise ValueError("Invalid checkpoint interval or selection range")
    if "attitude_correction" in config:
        if not isinstance(config["attitude_correction"], dict):
            raise ValueError("attitude_correction must be a mapping")
        from .attitude_correction import filter_parameters
        filter_parameters(config["attitude_correction"])
    return deepcopy(config)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
