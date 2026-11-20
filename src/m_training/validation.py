# SPDX-License-Identifier: GPL-3.0-or-later
"""Complete held-out validation and causal sequence-05 trajectories."""
import time
import numpy as np
import torch
from .data import pack
from .geometry import accumulate_poses, normalize_first_pose, trajectory_in_cam0
from .loss import axis_objective, AXES
from .runtime import capture_rng, restore_rng, to_device


def trajectory_result(relative, sequence):
    prediction = normalize_first_pose(trajectory_in_cam0(accumulate_poses(relative), sequence.extrinsic))
    aligned = (sequence.gt[0] @ np.linalg.inv(prediction[0])) @ prediction
    ate = float(np.sqrt(np.mean(np.sum((aligned[:, :3, 3] - sequence.gt[:, :3, 3]) ** 2, axis=1))))
    return prediction, {"ate_first_pose_m": ate}


@torch.inference_mode()
def validate(model, sequence, batches, config, device, progress=None):
    state, was_training = capture_rng(), model.training
    start = time.perf_counter()
    relative = np.full((len(sequence.frames), 6), np.nan, dtype=np.float32)
    length = config["data"]["window_length"]
    weights = torch.tensor(config["training"]["axis_weights"], device=device, dtype=torch.float32)
    sums, seen = {}, 0
    model.eval()
    try:
        for end in range(1, length):
            batch = to_device(pack(sequence, [0], end, targets=False), device)
            relative[end - 1] = model(batch)[0, -1].cpu().numpy()
        for index in range(len(batches)):
            cpu = batches[index]
            batch = to_device(cpu, device, pin=config["execution"]["pin_memory"])
            pred = model(batch)
            loss, parts = axis_objective(pred, batch["target"], weights)
            n = cpu["batch_size"]
            seen += n
            for key, value in {"loss": loss, **parts}.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            relative[np.asarray(cpu["starts"]) + length - 1] = pred[:, -1].cpu().numpy()
            if progress:
                progress(index + 1, len(batches), seen)
        expected = config["data"]["expected_counts"] or {"validation_windows": len(sequence.starts), "validation_batches": len(sequence.groups)}
        if seen != expected["validation_windows"] or len(batches) != expected["validation_batches"] or not np.isfinite(relative).all():
            raise ValueError("Incomplete or nonfinite sequence-05 validation")
        prediction, metrics = trajectory_result(relative, sequence)
        result = {f"validation_{k}": v / seen for k, v in sums.items()}
        if not all(np.isfinite(v) for v in result.values()) or not np.isfinite(metrics["ate_first_pose_m"]):
            raise FloatingPointError("Nonfinite validation loss or trajectory")
        result.update(validation_windows=seen, validation_batches=len(batches), validation_metrics=metrics,
                      validation_seconds=time.perf_counter() - start,
                      validation_rotation_loss=sum(result[f"validation_{axis}_loss"] for axis in AXES[:3]),
                      validation_translation_loss=sum(result[f"validation_{axis}_loss"] for axis in AXES[3:]))
        return result, {"prediction": prediction, "ground_truth": sequence.gt, "relative_prediction": relative}
    finally:
        model.train(was_training)
        restore_rng(state)
