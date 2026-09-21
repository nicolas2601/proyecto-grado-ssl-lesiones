"""Environment check for the cluster: versions, GPUs, bf16, /dev/shm, disk."""

from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys


def _version_of(module_name: str) -> str | None:
    try:
        return getattr(importlib.import_module(module_name), "__version__", "unknown")
    except ImportError:
        return None


def _gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "gpus": []}
    available = torch.cuda.is_available()
    gpus = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append({"name": props.name, "memory_gb": round(props.total_memory / 1024**3, 1)})
    return {
        "cuda_available": available,
        "cuda_version": torch.version.cuda,
        "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
        "nccl_available": bool(available and torch.distributed.is_nccl_available()),
        "gpus": gpus,
    }


def _shm_free_mb() -> float | None:
    try:
        return round(shutil.disk_usage("/dev/shm").free / 1024**2, 1)
    except OSError:
        return None


def collect_environment(work_dir: str = ".") -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _version_of("numpy"),
        "scipy": _version_of("scipy"),
        "pillow": _version_of("PIL"),
        "torch": _version_of("torch"),
        "torchvision": _version_of("torchvision"),
        "shm_free_mb": _shm_free_mb(),
        "work_dir_free_gb": round(shutil.disk_usage(work_dir).free / 1024**3, 1),
    }
    info.update(_gpu_info())
    return info


def main() -> None:
    print(json.dumps(collect_environment(), indent=2))


if __name__ == "__main__":
    main()
