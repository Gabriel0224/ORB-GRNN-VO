# SPDX-License-Identifier: GPL-3.0-or-later
import torch
AXES=("yaw","pitch","roll","tx","ty","tz")

def axis_objective(prediction, target, weights):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 6:
        raise ValueError("Expected matching [windows, time, 6] poses")
    error = prediction.float() - target.float()
    mse = error.square().mean(dim=(0, 1))
    parts = mse * weights
    bias = error.detach().mean(dim=(0, 1))
    diagnostics = {f"{axis}_{key}": vector[index] for key, vector in
                   (("loss", parts.detach()), ("mse", mse.detach()), ("bias", bias))
                   for index, axis in enumerate(AXES)}
    return parts.sum(), diagnostics
