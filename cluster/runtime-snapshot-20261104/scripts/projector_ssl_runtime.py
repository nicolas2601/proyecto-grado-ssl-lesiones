"""Runtime-only hooks around the registered frozen loop; never edits its source."""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
from unittest.mock import patch

import projector_hooks
from run_overlay_ssl import (CoverageState, LocalScalarTracker, bind_sampling_ids, digest,
                             finite_gradient_norm, validate_output_dir,
                             validate_wrapper_config)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"


def commit_pending(coverage, pending, step, batch_size):
    if step == coverage.steps:
        return
    if step != coverage.steps + 1 or len(pending) != batch_size:
        raise ValueError("Actual batch accesses do not match completed optimizer steps")
    coverage.record_successful_batch(pending)
    pending.clear()


def config_seal(config):
    config = {**config, "base_config": dict(config["base_config"])}
    config.pop("resume", None)
    config["base_config"].pop("resume_checkpoint", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def run(config_path):
    started = time.monotonic()
    config = validate_wrapper_config(json.loads(Path(config_path).read_text()))
    OBJECTIVE = config["objective"]
    for path, expected in ((config["manifest"], config["manifest_sha256"]),
                           (config["overlay_receipt"], config["overlay_receipt_sha256"])):
        if digest(path) != expected:
            raise ValueError("Input provenance mismatch")
    rows = [json.loads(line) for line in Path(config["manifest"]).read_text().splitlines()]
    if not rows or any(row["role"] not in {"source_fit", "pad_adapt"} for row in rows):
        raise ValueError("Unexpected training roles")
    receipt_path = Path(config["overlay_receipt"])
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "COMPLETE":
        raise ValueError("Overlay receipt is not complete")
    approved = set()
    for role in ("source_fit", "pad_adapt"):
        path = receipt_path.parent / f"{role}.jsonl"
        if digest(path) != receipt["output_sha256"][path.name]:
            raise ValueError("Derived role manifest changed")
        approved.update((role, row["image_id"], row["image_path"], row["identity_component_id"])
                        for row in map(json.loads, path.read_text().splitlines()))
    actual = {(r["role"], r["image_id"], r["image_path"], r["identity_component_id"]) for r in rows}
    if actual != approved or len(rows) != len(approved):
        raise ValueError("Training pool must exactly match the approved derived roles")
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate physical training paths")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training identities")
    base = config["base_config"]
    out = Path(base["output_dir"])
    if not out.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Run output must remain in the isolated experiment")
    validate_output_dir(out, config.get("resume", False))
    frozen = Path(json.loads((ROOT / "config/paths.json").read_text())["frozen_experiment_root"])
    sys.path[:0] = [str(frozen / "code/src")]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from dermssl.pretrain import config as config_mod, torch_loop, torch_simsiam
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Exactly one visible CUDA worker is required")
    if digest(base["initial_weights"]) != PUBLIC_SHA:
        raise ValueError("Public initializer changed")
    cache = json.loads((frozen / "evidence/torchhub-cache.json").read_text())
    if len(cache["cached_repos"]) != 1:
        raise ValueError("Expected one registered local torchhub checkout")
    cfg = config_mod.PretrainConfig(**base)
    config_mod.validate(cfg)
    if (cfg.method != "simsiam_dinov2" or cfg.backbone != "dinov2_vitb14"
            or cfg.freeze_blocks != 8 or cfg.anchor_distill != 1.0
            or cfg.anchor_l2sp != 0.0001 or cfg.batch_size_per_gpu != 16
            or cfg.clinical_batch_quota not in (1, 4) or cfg.max_steps not in (40, 6000)
            or cfg.seed not in (42, 43, 44) or cfg.precision != "bf16"
            or cfg.head_warmup_steps != 32 or cfg.num_workers != 0
            or (cfg.clinical_batch_quota == 1 and config["lr_multiplier"] != 1)):
        raise ValueError("Configuration is outside the authorized isolated experiment")
    seal = config_seal(config)
    resume_path = Path(cfg.resume_checkpoint) if config.get("resume") else None
    ids = [row["image_id"] for row in rows]
    domains = [int(row["role"] == "pad_adapt") for row in rows]
    coverage = CoverageState(ids, domains)
    if resume_path:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state.get("wrapper_sha256") != seal:
            raise ValueError("Resume recipe/overlay differs from sealed run")
        coverage.restore(state["actual_coverage"])
        if coverage.steps != state["step"]:
            raise ValueError("Resume coverage is not aligned to model state")
        del state
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path): digest(path) for path in
                     sorted((frozen / "code/src/dermssl/pretrain").glob("*.py"))}
    source_hashes.update({str(Path(__file__)): digest(__file__),
                         str(ROOT / "scripts/run_overlay_ssl.py"): digest(ROOT / "scripts/run_overlay_ssl.py")})
    (out / "launch-seal.json").write_text(json.dumps({"wrapper_sha256": seal,
        "config": config, "source_sha256": source_hashes,
        "deterministic_algorithms": True}, indent=2))
    pending = []
    original_build, original_save = torch_loop._build, torch_loop._save_checkpoint
    original_lr, original_clip = config_mod.effective_lr, torch_loop._clip_gradients
    original_sampling = torch_loop.sampling_contract.load_sampling_plan
    tracker = LocalScalarTracker(out / "scalars.jsonl")

    class AccessedDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            result = self.dataset[index]
            pending.append(int(index))
            return result

    def build(*args, **kwargs):
        parts = list(original_build(*args, **kwargs))
        parts[3] = AccessedDataset(parts[3])
        return tuple(parts)

    def local_backbone(name, pretrained):
        model = torch.hub.load(cache["cached_repos"][0], name, source="local", pretrained=False)
        return model, model.embed_dim

    def save(*args, **kwargs):
        commit_pending(coverage, pending, kwargs["step"], cfg.batch_size_per_gpu)
        kwargs["position"] = {**kwargs.get("position", {}),
                              "actual_coverage": coverage.checkpoint_payload(),
                              "wrapper_sha256": seal}
        return original_save(*args, **kwargs)

    def callback(model, step):
        commit_pending(coverage, pending, step, cfg.batch_size_per_gpu)
        if step % 500 == 0:
            print(json.dumps({"step": step, "actual_unique_ids": sum(v > 0 for v in coverage.counts.values()),
                              "domain_draws": coverage.domain_counts}), flush=True)
        return {}

    def timeout(signum, frame):
        raise TimeoutError("Allocated training GPU-hour limit reached")

    previous_handler = signal.signal(signal.SIGTERM, timeout)
    remaining = config["training_gpu_hours_limit"] * 3600 - (time.monotonic() - started) - 5
    if remaining <= 0:
        raise TimeoutError("Setup exhausted training budget")
    timer = threading.Timer(remaining, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    status, final = "FAILED", None
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        with patch.object(config_mod, "collect_images", return_value=[Path(r["image_path"]) for r in rows]), \
             patch.object(torch_loop.sampling_contract, "load_sampling_plan", side_effect=lambda p, s, images: original_sampling(p, s, bind_sampling_ids(rows, images))), \
             patch.object(config_mod, "effective_lr", side_effect=lambda *a: original_lr(*a) * config["lr_multiplier"]), \
             patch.object(torch_loop, "_build", side_effect=build), \
             patch.object(torch_simsiam, "SimSiam", new=projector_hooks.make_projector_simsiam(torch_simsiam.SimSiam)), \
             patch.object(torch_simsiam, "simsiam_loss", new=projector_hooks.make_projector_loss(OBJECTIVE)), \
             patch.object(torch_simsiam, "_build_backbone", side_effect=local_backbone), \
             patch.object(torch_loop, "_save_checkpoint", side_effect=save), \
             patch.object(torch_loop, "_clip_gradients", side_effect=lambda *a: finite_gradient_norm(original_clip(*a))), \
             patch.object(torch_loop.tracking, "build_tracker", return_value=tracker):
            final = torch_loop.train(cfg, step_callback=callback)
        if coverage.steps != cfg.max_steps or pending:
            raise ValueError("Partial training cannot be declared COMPLETE")
        if any(digest(path) != value for path, value in source_hashes.items()):
            raise ValueError("Imported source changed during training")
        status = "COMPLETE"
    finally:
        timer.cancel()
        signal.signal(signal.SIGTERM, previous_handler)
        seconds = time.monotonic() - started
        result = {"status": status, "steps": coverage.steps, "allocated_gpu_hours": seconds / 3600,
                  "wrapper_sha256": seal, "backbone": str(final) if final else None,
                  "backbone_sha256": digest(final) if final else None,
                  "coverage": coverage.checkpoint_payload()}
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({key: value for key, value in result.items() if key != "coverage"}), flush=True)
    return result


if __name__ == "__main__":
    _cfg = json.loads(Path(sys.argv[1]).read_text())
    if _cfg.get("objective") == "icone":
        import icone_ssl_runtime
        icone_ssl_runtime.run(sys.argv[1])
    elif _cfg.get("objective") == "maskfeat":
        import maskfeat_runtime
        maskfeat_runtime.run(sys.argv[1])
    elif _cfg.get("objective") == "msn":
        import msn_runtime
        msn_runtime.run(sys.argv[1])
    elif _cfg.get("objective") == "swav":
        import swav_runtime
        swav_runtime.run(sys.argv[1])
    elif _cfg.get("objective") == "diet":
        import diet_runtime
        diet_runtime.run(sys.argv[1])
    else:
        run(sys.argv[1])
