# SPDX-License-Identifier: GPL-3.0-or-later
"""Atomic artifacts, reproducibility and exact checkpoint state."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile
import numpy as np
import torch


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(root.glob("*.py"))}


def configure(seed, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    _atomic(path, lambda f: f.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()))


def atomic_checkpoint(path, value):
    _atomic(path, lambda f: torch.save(value, f))


def atomic_npz(path, **values):
    _atomic(path, lambda f: np.savez_compressed(f, **values))


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def acquire_lock(path):
    stream = Path(path).open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another process owns this training directory") from None
    return stream


def to_device(batch, device, pin=False):
    def transfer(value):
        if not torch.is_tensor(value):
            return value
        if pin and device.type == "cuda" and value.device.type == "cpu" and not value.is_pinned():
            value = value.pin_memory()
        return value.to(device, non_blocking=device.type == "cuda")
    return {k: transfer(v) for k, v in batch.items()}
