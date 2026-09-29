"""The execution environment behind a result, recorded next to the result.

v7.1 recorded no versions and no thread settings, so when a T4 reproduction
of its grid drifted by ~1e-3 nothing in the JSON could say whether a library,
a thread count or a cuDNN flag had changed. Every script that writes an
evaluation JSON calls :func:`write_sidecar` after writing it.

The block lives in a SIDECAR, ``<name>.environment.json``, never inside the
result: guards pin results and manifests by the sha256 of their bytes, and a
block that differs per machine inside those bytes would make the same payload
hash differently on a T4 and on the cluster. The sidecar names the file it
describes and its sha256, so the pair stays linked.

Pure on purpose: torch is never imported here (only ``torch_*`` modules may
import it). If the script already loaded torch, its module object is read from
``sys.modules``; if it did not, torch cannot have influenced the result and the
block says so. GPU names are only read when CUDA is already initialised, so a
CPU-only script never creates a CUDA context just to describe itself.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

SCHEMA = 1
SIDECAR_SUFFIX = ".environment.json"
# Distribution name -> importable module name.
PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn", "torch": "torch",
    "torchvision": "torchvision", "joblib": "joblib", "threadpoolctl": "threadpoolctl",
}
ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")
# Parallelism flags a script may expose; whichever it has are recorded as given.
KNOBS = ("probe_jobs", "boot_jobs", "decode_workers", "num_workers", "workers", "n_jobs", "jobs",
         "batch_size", "threads")
THREADPOOL_KEYS = ("user_api", "internal_api", "num_threads", "version")


def package_versions(modules=None) -> dict:
    """Version of each package: the loaded module's ``__version__``, else the installed one, else None."""
    modules = sys.modules if modules is None else modules
    return {dist: _version(dist, modules.get(module)) for dist, module in PACKAGES.items()}


def _version(dist: str, module) -> str | None:
    loaded = getattr(module, "__version__", None)
    if loaded is not None:
        return str(loaded)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_state(torch) -> dict:
    """cuDNN/CUDA/determinism/threads of an already loaded torch; ``{"loaded": False}`` otherwise."""
    if torch is None:
        return {"loaded": False}
    cudnn = torch.backends.cudnn
    return {
        "loaded": True, "version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn.version() if cudnn.is_available() else None,
        "cudnn_deterministic": bool(cudnn.deterministic), "cudnn_benchmark": bool(cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "num_threads": int(torch.get_num_threads()), **_cuda_devices(torch.cuda),
    }


def _cuda_devices(cuda) -> dict:
    available = bool(cuda.is_available())
    initialized = bool(available and cuda.is_initialized())
    gpus = [{"index": i, "name": cuda.get_device_name(i)} for i in range(cuda.device_count())] if initialized else []
    return {"cuda_available": available, "cuda_initialized": initialized, "gpus": gpus}


def threadpools() -> list | None:
    """Loaded BLAS/OpenMP pools as threadpoolctl sees them; None when threadpoolctl is absent."""
    try:
        from threadpoolctl import threadpool_info  # noqa: PLC0415
    except ImportError:
        return None
    return [{key: pool.get(key) for key in THREADPOOL_KEYS} for pool in threadpool_info()]


def knobs(args=None, **extra) -> dict:
    """The script's parallelism flags (only those it defines) plus any explicit extras."""
    found = {name: getattr(args, name) for name in KNOBS if hasattr(args, name)}
    found.update(extra)
    return found


def snapshot(args=None, *, modules=None, environ=None, cpu_count=None, pools=None, **extra) -> dict:
    """The ``environment`` block: every argument after ``args`` is injectable for tests."""
    modules = sys.modules if modules is None else modules
    environ = os.environ if environ is None else environ
    return {
        "schema": SCHEMA,
        "python": platform.python_version(),
        "packages": package_versions(modules),
        "torch": torch_state(modules.get("torch")),
        "env": {name: environ.get(name) for name in ENV_VARS},
        "threadpools": threadpools() if pools is None else pools,
        "cpu_count": os.cpu_count() if cpu_count is None else cpu_count,
        "knobs": knobs(args, **extra),
    }


def sidecar_path(path) -> Path:
    """``dir/bank.json`` -> ``dir/bank.environment.json``."""
    path = Path(path)
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(stem + SIDECAR_SUFFIX)


def write_sidecar(path, args=None, *, environment=None, **extra) -> Path:
    """Write the block next to an already written result, naming it and its sha256.

    Call it AFTER the result is on disk: the sidecar pins the exact bytes it
    describes. ``environment`` is injectable for tests; otherwise :func:`snapshot`.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no result to describe at {path}")
    record = {"describes": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "environment": snapshot(args, **extra) if environment is None else environment}
    target = sidecar_path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target
