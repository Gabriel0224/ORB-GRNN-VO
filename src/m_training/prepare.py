# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify frozen inputs and create portable CPU training batches."""
import argparse
import json
from pathlib import Path
import torch
from .config import config_hash, load_config
from .data import Sequence, pack
from .runtime import acquire_lock, atomic_checkpoint, atomic_json, configure, sha256


def data_signature(config):
    return config_hash({"data": config["data"], "batch_size": config["training"]["batch_size"],
                        "seed": config["seed"], "epochs": config["training"]["max_epochs"]})


def prepare(config, inputs, output, inspect_only=False):
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    if output == inputs or output.is_relative_to(inputs):
        raise ValueError("Prepared batches must be stored outside read-only inputs")
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(output / ".prepare.lock")
    try:
        if (output / "manifest.json").exists():
            raise FileExistsError("Prepared manifest already exists; use it or choose a new directory")
        records = {"train": [], "validation": []}
        audit, mapping, input_hashes = {}, {}, {}
        for sequence in config["data"]["train_sequences"] + [config["data"]["validation_sequence"]]:
            q = Sequence(inputs, sequence, config, neighbors=not inspect_only)
            split = "train" if sequence in config["data"]["train_sequences"] else "validation"
            input_hashes.update(q.file_hashes)
            audit[sequence] = {"pairs": len(q.frames), "windows": len(q.starts), "batches": len(q.groups),
                               "empty_pairs": sum(len(x) == 0 for x in q.frames)}
            mapping[sequence] = {"starts": q.starts.tolist(), "groups": q.groups}
            for index, starts in enumerate(q.groups):
                row = {"sequence": sequence, "windows": len(starts), "starts": starts}
                if not inspect_only:
                    batch = pack(q, starts, config["data"]["window_length"])
                    name = f"{split}/seq{sequence}_group{index:03d}.pt"
                    dest = output / name
                    if dest.exists():
                        raise FileExistsError(f"Refusing to replace an existing batch: {name}")
                    atomic_checkpoint(dest, batch)
                    row.update(path=name, sha256=sha256(dest), points=len(batch["x"]))
                records[split].append(row)
            print(f"Verified sequence {sequence}: {len(q.starts)} windows, {len(q.groups)} batches", flush=True)
        counts = {f"{split}_windows": sum(r["windows"] for r in records[split]) for split in records}
        counts.update({f"{split}_batches": len(records[split]) for split in records})
        if config["data"]["expected_counts"] is not None and counts != config["data"]["expected_counts"]:
            raise ValueError(f"Frozen-input count mismatch: {counts}")
        if any(value < 1 for value in counts.values()):
            raise ValueError("Training and validation each need at least one complete window")
        atomic_json(output / "input_audit.json", {"sequences": audit, "counts": counts,
                    "files": input_hashes, "data_signature": data_signature(config), "passed": True})
        atomic_json(output / "window_mapping.json", mapping)
        generator = torch.Generator().manual_seed(config["seed"])
        orders = [torch.randperm(counts["train_batches"], generator=generator).tolist()
                  for _ in range(config["training"]["max_epochs"])]
        atomic_json(output / "data_order.json", orders)
        if not inspect_only:
            atomic_json(output / "manifest.json", {"schema": "m-prepared-v1", "data_signature": data_signature(config),
                        "counts": counts, "records": records, "input_hashes": input_hashes,
                        "data_order_sha256": sha256(output / "data_order.json")})
        return counts
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", required=True, help="Read-only P1 candidates and reference folder")
    parser.add_argument("--prepared", required=True, help="New directory for CPU packs and input audit")
    parser.add_argument("--inspect-only", action="store_true", help="Verify hashes, windows and grouping without writing batch tensors")
    args = parser.parse_args()
    config = load_config(args.config)
    # CPU preparation never calls CUDA and uses an independent order generator.
    torch.set_num_threads(config["execution"]["cpu_threads"])
    prepare(config, args.inputs, args.prepared, args.inspect_only)


if __name__ == "__main__":
    main()
