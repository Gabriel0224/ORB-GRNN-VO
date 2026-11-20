# SPDX-License-Identifier: GPL-3.0-or-later
"""Train GRNN-VO with validation and checkpoint recovery."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import time
import traceback
import numpy as np
import torch
from .config import load_config
from .data import Sequence, PreparedBatches, contained
from .loss import axis_objective
from .model import PointSetModel
from .prepare import data_signature
from .runtime import (acquire_lock, append_json, atomic_checkpoint, atomic_json, atomic_npz,
                      capture_rng, configure, now, restore_rng, sha256, source_hashes, to_device, _atomic)
from .schedule import V3ValidationStaircase
from .validation import validate


STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def build_model(config):
    e = config["execution"]
    cls = PointSetModel
    if e["implementation"] == "compact":
        from .execution import bind
        cls = bind(cls, chunk=e["frame_chunk_size"], retain_nodes=e["retain_nodes"],
                   graph_checkpoint=e["graph_checkpointing"], compact_graph=True)
    return cls(frame_chunk_size=e["frame_chunk_size"], checkpoint_enabled=e["activation_checkpointing"])


def stopping_reason(epoch, scheduler, max_epochs):
    if scheduler.stage == len(scheduler.state_dict()["rates"]) - 1 and scheduler.num_bad_epochs >= 20:
        return "minimum_lr_20_non_improving_full_validations"
    return "max_epochs" if epoch >= max_epochs else None


def repair_logs(output, state):
    """Archive log entries after the checkpoint's committed step."""
    checkpoint_steps = state["global_step"]
    history = state["history"]
    expected_epochs = state["epoch"] if state["epoch_complete"] else state["epoch"] - 1
    if [r["epoch"] for r in history] != list(range(1, expected_epochs + 1)):
        raise ValueError("Checkpoint history is incomplete")
    step_path = output / "steps.jsonl"
    raw = step_path.read_bytes() if step_path.exists() else b""
    kept = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row["global_step"] <= checkpoint_steps:
            kept.append(row)
    if [r["global_step"] for r in kept] != list(range(1, checkpoint_steps + 1)):
        raise ValueError("Step log before the checkpoint is incomplete; restore a verified backup")
    for path, rows in [(step_path, kept), (output / "training.jsonl", history)]:
        data = ("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)).encode()
        old = path.read_bytes() if path.exists() else b""
        if old != data:
            if old:
                backup = output / "recovery" / str(time.time_ns()) / path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(old)
            _atomic(path, lambda f, data=data: f.write(data))


def run(config, inputs, prepared, output, device, resume=False):
    global STOP
    STOP = False
    inputs, prepared, output = [Path(p).resolve() for p in (inputs, prepared, output)]
    if any(output == p or output.is_relative_to(p) for p in [inputs, prepared]):
        raise ValueError("Training output must be separate from inputs and prepared batches")
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output / ".run.lock")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        _run_locked(config, inputs, prepared, output, torch.device(device), resume)
    except BaseException as error:
        atomic_json(output / "failure.json", {"timestamp": now(), "type": type(error).__name__, "error": traceback.format_exc()})
        raise
    finally:
        lock.close()


def _run_locked(config, inputs, prepared, output, device, resume):
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This implementation supports CPU and CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a matching PyTorch build")
        torch.cuda.set_device(device)
    checkpoint_path = output / "last.pt"
    if resume != checkpoint_path.exists():
        raise ValueError("Use a new output directory, or explicitly resume its existing last.pt")
    if not resume and any(p.name != ".run.lock" for p in output.iterdir()):
        raise FileExistsError("Fresh training requires an empty output directory")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["schema"] != "m-prepared-v1" or manifest["data_signature"] != data_signature(config):
        raise ValueError("Prepared data/configuration mismatch")
    if config["data"]["expected_counts"] is not None and manifest["counts"] != config["data"]["expected_counts"]:
        raise ValueError("Prepared training/validation counts mismatch")
    for name, digest in manifest["input_hashes"].items():
        if sha256(contained(inputs, name)) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    order_hash = sha256(prepared / "data_order.json")
    if order_hash != manifest["data_order_sha256"]:
        raise ValueError("Saved data order checksum mismatch")
    orders = json.loads((prepared / "data_order.json").read_text())
    count = manifest["counts"]["train_batches"]
    if len(orders) != config["training"]["max_epochs"] or any(sorted(o) != list(range(count)) for o in orders):
        raise ValueError("Data order is not one complete permutation per epoch")
    e = config["execution"]
    configure(config["seed"], e["cpu_threads"])
    train = PreparedBatches(prepared, manifest["records"]["train"], e["implementation"], e["batch_cache_size"])
    val = PreparedBatches(prepared, manifest["records"]["validation"], "reference", e["batch_cache_size"])
    q = Sequence(inputs, config["data"]["validation_sequence"], config)
    model = build_model(config).to(device).train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != config["model"]["parameters"]:
        raise ValueError(f"Unexpected model parameter count: {parameters}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = V3ValidationStaircase(opt)
    sources = source_hashes()
    environment = {"torch": str(torch.__version__), "numpy": str(np.__version__), "python": platform.python_version(),
                   "device_type": device.type, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "threads": torch.get_num_threads(), "precision": "FP32", "TF32": False}
    state = {"epoch": 1, "epoch_complete": False, "next_batch_index": 0, "global_step": 0,
             "epoch_sums": {}, "windows_seen": 0, "epoch_seconds": 0.0, "elapsed_compute_seconds": 0.0,
             "best_validation_loss": float("inf"), "best_validation_epoch": 0,
             "best_origin_ate": float("inf"), "best_origin_epoch": 0, "history": [], "stopping_reason": None}
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key, value in [("config", config), ("source_hashes", sources), ("data_order_sha256", order_hash),
                           ("input_hashes", manifest["input_hashes"]), ("environment", environment)]:
            if saved[key] != value:
                raise ValueError(f"Exact resume rejected: {key} differs")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        opt.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        state = saved["progress"]
        restore_rng(saved["rng_state"])
        repair_logs(output, state)
    else:
        atomic_json(output / "config.json", config)
        atomic_json(output / "run.json", {"started_at": now(), "parameters": parameters, "environment": environment,
                    "source_hashes": sources, "data_order_sha256": order_hash, "input_hashes": manifest["input_hashes"]})
        atomic_json(output / "data_order.json", orders)

    def save(name):
        atomic_checkpoint(output / name, {"format": "m-training/v1", "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng(), "epoch": state["epoch"], "global_step": state["global_step"],
            "epoch_complete": state["epoch_complete"], "progress": state, "config": config,
            "source_hashes": sources, "input_hashes": manifest["input_hashes"], "data_order_sha256": order_hash,
            "environment": environment, "saved_at": now()})

    def status(stage, **fields):
        atomic_json(output / "status.json", {"stage": stage, "epoch": state["epoch"],
                    "batch": state["next_batch_index"], "global_step": state["global_step"],
                    "timestamp": now(), **fields})

    def commit_role(name, epoch):
        src = output / f"epoch_{epoch:03d}.pt"
        loaded = torch.load(src, map_location="cpu", weights_only=False)
        if loaded["epoch"] != epoch or not loaded["epoch_complete"]:
            raise ValueError("Selected checkpoint is not a complete epoch")
        model_state = loaded["model_state_dict"]
        if set(model_state) != set(model.state_dict()) or any(model_state[k].shape != v.shape for k, v in model.state_dict().items()):
            raise ValueError("Selected checkpoint is incompatible")
        _atomic(output / name, lambda f: f.write(src.read_bytes()))
        return {"epoch": epoch, "filename": name, "sha256": sha256(output / name)}

    if not resume:
        save("initial.pt")
        save("last.pt")
    weights = torch.tensor(config["training"]["axis_weights"], dtype=torch.float32, device=device)
    start_epoch = state["epoch"] + int(state["epoch_complete"])
    if state["epoch_complete"]:
        state.update(next_batch_index=0, epoch_sums={}, windows_seen=0, epoch_seconds=0.0)
    completed = bool(state["stopping_reason"])
    for epoch in ([] if completed else range(start_epoch, config["training"]["max_epochs"] + 1)):
        state.update(epoch=epoch, epoch_complete=False)
        model.train()
        wall_start = time.perf_counter()
        resumed_mid_epoch = state["next_batch_index"] > 0
        lr = float(opt.param_groups[0]["lr"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch_index in range(state["next_batch_index"], len(train)):
            if STOP:
                save("last.pt")
                status("paused")
                return
            begin = time.perf_counter()
            batch = to_device(train[orders[epoch - 1][batch_index]], device, e["pin_memory"])
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            loss, parts = axis_objective(pred, batch["target"], weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, batch {batch_index + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - begin
            n = batch["batch_size"]
            state["global_step"] += 1
            state["next_batch_index"] = batch_index + 1
            state["windows_seen"] += n
            state["epoch_seconds"] += dt
            state["elapsed_compute_seconds"] += dt
            for key, value in {"loss": loss.detach(), **parts}.items():
                state["epoch_sums"][key] = state["epoch_sums"].get(key, 0.0) + float(value) * n
            row = {"epoch": epoch, "batch": batch_index + 1, "global_step": state["global_step"],
                   "learning_rate": lr, "loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                   "windows_seen": state["windows_seen"], "seconds": dt, "timestamp": now()}
            append_json(output / "steps.jsonl", row)
            status("training", loss=row["loss"], learning_rate=lr, batches=len(train))
            if state["global_step"] % config["saving"]["last_every_steps"] == 0:
                save("last.pt")
            del pred, loss, parts, batch
        if state["global_step"] != epoch * len(train) or state["windows_seen"] != manifest["counts"]["train_windows"]:
            raise ValueError("Incomplete training epoch")
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        save("last.pt")  # Recover before the validation/scheduler transaction if interrupted.
        validation, trajectory = validate(model, q, val, config, device,
            progress=lambda index, count, seen: status("validation", validation_batch=index, validation_batches=count, validation_windows=seen))
        scheduler.step(validation["validation_loss"], validation_complete=True)
        reason = stopping_reason(epoch, scheduler, config["training"]["max_epochs"])
        loss_improved = validation["validation_loss"] < state["best_validation_loss"]
        ate = validation["validation_metrics"]["ate_first_pose_m"]
        ate_improved = ate < state["best_origin_ate"]
        if loss_improved:
            state.update(best_validation_loss=validation["validation_loss"], best_validation_epoch=epoch)
        if ate_improved:
            state.update(best_origin_ate=ate, best_origin_epoch=epoch)
        state["epoch_seconds"] += validation["validation_seconds"]
        state["elapsed_compute_seconds"] += validation["validation_seconds"]
        row = {"epoch": epoch, "global_step": state["global_step"], "learning_rate": lr,
               "next_learning_rate": float(opt.param_groups[0]["lr"]), "train_windows": state["windows_seen"],
               "train_batches": len(train), **{f"train_{k}": v / state["windows_seen"] for k, v in state["epoch_sums"].items()},
               **validation, "epoch_compute_seconds": state["epoch_seconds"],
               "epoch_wall_seconds": time.perf_counter() - wall_start, "epoch_wall_seconds_exact": not resumed_mid_epoch,
               "elapsed_compute_seconds": state["elapsed_compute_seconds"], "peak_training_allocated_bytes": peak,
               "peak_training_reserved_bytes": reserved, "best_validation_epoch": state["best_validation_epoch"],
               "best_validation_loss": state["best_validation_loss"], "best_origin_epoch": state["best_origin_epoch"],
               "best_origin_ate": state["best_origin_ate"], "scheduler": scheduler.state_dict(), "stopping_reason": reason,
               "timestamp": now()}
        atomic_npz(output / "trajectories" / f"epoch_{epoch:03d}.npz", **trajectory)
        atomic_json(output / "trajectories" / f"epoch_{epoch:03d}.json", validation)
        state.update(epoch_complete=True, stopping_reason=reason)
        state["history"].append(row)
        save(f"epoch_{epoch:03d}.pt")
        append_json(output / "training.jsonl", row)
        save("last.pt")
        if loss_improved:
            commit_role("best_loss_all.pt", epoch)
        if ate_improved:
            commit_role("best_ate_origin_all.pt", epoch)
        status("epoch_complete", **validation, learning_rate=lr)
        print(f"epoch={epoch} train={row['train_loss']:.8f} validation={validation['validation_loss']:.8f} "
              f"05_origin_ate={ate:.6f} lr={lr:g}", flush=True)
        if reason:
            break
        if STOP:
            status("paused")
            return
        state.update(next_batch_index=0, epoch_sums={}, windows_seen=0, epoch_seconds=0.0)
    history = state["history"]
    if not state["stopping_reason"] or not state["epoch_complete"]:
        raise RuntimeError("Training ended without a complete stopping epoch")
    final_epoch = history[-1]["epoch"]
    candidates = history[-config["saving"]["selection_last_epochs"]:]
    best_loss = min(candidates, key=lambda r: r["validation_loss"])
    best_ate = min(candidates, key=lambda r: r["validation_metrics"]["ate_first_pose_m"])
    roles = {"best_loss_last50": commit_role("best_loss_last50.pt", best_loss["epoch"]),
             "best_ate_origin_last50": commit_role("best_ate_origin_last50.pt", best_ate["epoch"]),
             "final": commit_role("final.pt", final_epoch),
             "best_loss_all": commit_role("best_loss_all.pt", state["best_validation_epoch"]),
             "best_ate_origin_all": commit_role("best_ate_origin_all.pt", state["best_origin_epoch"])}
    atomic_json(output / "selection.json", {"range": [candidates[0]["epoch"], final_epoch], "roles": roles,
                "loss_rule": "minimum complete validation loss", "ate_rule": "minimum raw initial-pose-aligned validation ATE",
                "strict_ties_keep_earlier": True, "test_used": False})
    atomic_json(output / "completion.json", {"completed": True, "epochs": final_epoch,
                "full_validations": len(history), "global_step": state["global_step"], "roles": roles,
                "parameters": parameters, "stopping_reason": state["stopping_reason"], "timestamp": now()})
    status("completed", stop_epoch=final_epoch, roles=roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Restore this output directory's last.pt, including optimizer and RNG")
    args = parser.parse_args()
    run(load_config(args.config), args.inputs, args.prepared, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
