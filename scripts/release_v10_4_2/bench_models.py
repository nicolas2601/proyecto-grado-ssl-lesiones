"""Inference cost of the VERA models (ONNX Runtime): latency, load time and memory per model.

  python bench_models.py --models-dir DIR --out OUT.json --threads 2,4,8,16 [--gpu]

Each (model, provider, threads) case runs in its own subprocess so peak RSS belongs to that case only.
Inputs are fixed random tensors: these graphs do the same work for any pixel content.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODELS = {
    "classifier_v10_4_0_d4": ("classifier_v10_4_0_d4.onnx", (1, 3, 448, 448)),
    "classifier_v10_5_0_single": ("classifier_v10_5_0_single.onnx", (1, 3, 448, 448)),
    "segmenter_v10_4_1_resnet34": ("segmenter_v10_4_1_resnet34.onnx", (1, 3, 224, 224)),
    "segmenter_v10_4_2_dpt": ("segmenter_v10_4_2_dpt.onnx", (1, 3, 448, 448)),
}
WARMUP = 3


def gpu_memory_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def run_case(models_dir: Path, key: str, provider: str, threads: int, runs: int) -> dict:
    import onnxruntime as ort

    file, shape = MODELS[key]
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    gpu_before = gpu_memory_mb()
    started = time.perf_counter()
    session = ort.InferenceSession(str(models_dir / file), sess_options=options, providers=providers)
    load_s = time.perf_counter() - started
    feed = {session.get_inputs()[0].name: np.random.default_rng(0).standard_normal(shape).astype(np.float32)}
    for _ in range(WARMUP):
        session.run(None, feed)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t) * 1000)
    gpu_after = gpu_memory_mb()
    return {
        "model": key, "provider": provider, "active_providers": session.get_providers(), "threads": threads,
        "runs": runs, "load_s": round(load_s, 2), "p50_ms": round(float(np.median(times)), 1),
        "p90_ms": round(float(np.percentile(times, 90)), 1), "mean_ms": round(float(np.mean(times)), 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "gpu_mem_delta_mb": (gpu_after - gpu_before) if gpu_before is not None and gpu_after is not None else None,
        "model_mb": round((models_dir / file).stat().st_size / 1e6, 1),
    }


def hardware() -> dict:
    cpu = platform.processor()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    import onnxruntime as ort

    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "gpu": gpu, "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--threads", default="2,4,8")
    parser.add_argument("--cpu-runs", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--case", nargs=4, metavar=("MODEL", "PROVIDER", "THREADS", "RUNS"))
    args = parser.parse_args()
    if args.case:
        key, provider, threads, runs = args.case
        print(json.dumps(run_case(args.models_dir, key, provider, int(threads), int(runs))))
        return
    cases = [(key, "cpu", int(t), args.cpu_runs) for key in MODELS for t in args.threads.split(",")]
    if args.gpu:
        cases += [(key, "cuda", 1, 50) for key in MODELS]
    results = []
    for key, provider, threads, runs in cases:
        proc = subprocess.run([sys.executable, __file__, "--models-dir", str(args.models_dir), "--case", key,
                               provider, str(threads), str(runs)], capture_output=True, text=True)
        line = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 else None
        row = json.loads(line) if line else {"model": key, "provider": provider, "threads": threads,
                                             "error": proc.stderr.strip().splitlines()[-1:]}
        print(json.dumps(row), flush=True)
        results.append(row)
    report = {"hardware": hardware(), "results": results}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
