# SPDX-License-Identifier: GPL-3.0-or-later
"""One command from user images and supplied pose labels to trained M."""
import argparse
import copy
import json
from pathlib import Path
import traceback
from .config import config_hash, load_config
from .frontend import build
from .images import extract_dataset, load_dataset
from .prepare import prepare
from .runtime import acquire_lock, atomic_json, now, sha256


def inventory(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def pipeline(dataset, config_path, work, device="cuda:0", resume=False, jobs=2, prepare_only=False):
    from .train import run
    config = load_config(config_path)
    if config["data"].get("protocol") != "custom_images":
        raise ValueError("Use configs/m_images.yaml for user-provided images and pose labels")
    dataset = Path(dataset).resolve()
    records = load_dataset(dataset, require_labels=True)
    train = [k for k, v in records.items() if v["split"] == "train"]
    validation = [k for k, v in records.items() if v["split"] == "validation"]
    if not train or len(validation) != 1:
        raise ValueError("Supply training sequences and exactly one separate complete validation sequence")
    # Sequence names/counts are dataset metadata, not architecture hyperparameters.
    config = copy.deepcopy(config)
    config["data"].update(train_sequences=train, validation_sequence=validation[0], expected_counts=None)
    signature = {"dataset_sha256": sha256(dataset), "training_config": config_hash(config), "cache_policy": "keep"}
    work = Path(work).resolve()
    if not resume and work.exists():
        raise FileExistsError("Choose a new work directory, or explicitly use --resume")
    work.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(work / ".pipeline.lock")
    status_path = work / "pipeline.json"
    try:
        if resume:
            previous = json.loads(status_path.read_text())
            if previous["signature"] != signature:
                raise ValueError("Dataset or training configuration changed")
            if previous["stage"] == "completed":
                print(f"Already completed: {work / 'training'}")
                return
        atomic_json(work / "resolved_config.json", config)
        def status(stage, **fields):
            atomic_json(status_path, {"stage": stage, "signature": signature, "timestamp": now(), **fields})
        native_build = work / "native-build"
        inputs, prepared, training = work / "generated-inputs", work / "prepared", work / "training"
        if not (inputs / "manifest.json").exists():
            if inputs.exists():
                raise RuntimeError("Partial extraction retained. Inspect it and use a new work directory; it will not be overwritten")
            status("building_frontend")
            binary = build(native_build, jobs)
            status("extracting_images")
            extract_dataset(dataset, inputs, binary, config["seed"], with_labels=True)
        else:
            manifest = json.loads((inputs / "manifest.json").read_text())
            for name, item in records.items():
                if [sha256(p) for p in item["images"]] != manifest["sequences"][name]["image_sha256_in_order"]:
                    raise ValueError("Raw image content/order changed; frozen features cannot be silently reused")
                original_label_hashes = manifest["sequences"][name]["source_label_hashes"]
                if item["source_label_hashes"] != original_label_hashes:
                    raise ValueError("Supplied poses, timing or label validity/source boundaries changed")
        if not (prepared / "manifest.json").exists():
            status("preparing_batches")
            prepare(config, inputs, prepared)
        inventories = {"generated-inputs": inventory(inputs), "prepared": inventory(prepared)}
        atomic_json(work / "intermediate_inventory.json", inventories)
        atomic_json(work / "input_provenance.json", {
            "images": json.loads((inputs / "manifest.json").read_text()),
            "reference": json.loads((inputs / "reference" / "manifest.json").read_text()),
            "frontend_sources": json.loads((native_build / "build_manifest.json").read_text()),
            "prepared": json.loads((prepared / "manifest.json").read_text()),
        })
        status("prepared")
        if prepare_only:
            print(f"Images and labels prepared. Continue with the same command and --resume: {work}")
            return
        status("training")
        run(config, inputs, prepared, training, device, resume=(training / "last.pt").exists())
        if not (training / "completion.json").exists():
            status("paused", intermediates_retained_for_resume=True)
            return
        status("completed", training_output="training", cache_policy="keep",
               cache_files_retained=True)
        print(f"Completed: {training}", flush=True)
    except BaseException:
        atomic_json(work / "pipeline_failure.json", {"timestamp": now(), "error": traceback.format_exc(),
                    "intermediates_preserved": True, "automatic_retry": False})
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="YAML describing user images, intrinsics, timing and camera poses")
    parser.add_argument("--config", required=True, help="Use configs/m_images.yaml")
    parser.add_argument("--work-dir", required=True, help="New runtime workspace; generated inputs and prepared batches remain after training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after CPU image processing and batch preparation")
    args = parser.parse_args()
    pipeline(args.dataset, args.config, args.work_dir, args.device, args.resume, args.jobs, args.prepare_only)


if __name__ == "__main__":
    main()
