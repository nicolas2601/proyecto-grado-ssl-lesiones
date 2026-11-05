"""Runtime hooks for the paired tone contrast, built on top of overlay_ssl_runtime.

This is a SIBLING of ``overlay_ssl_runtime.py``, not an edit of it. That wrapper
produced the control arm ``ssl_overlay_A_s42_6000_v1`` and its source hash is
sealed into that run's receipt; changing it in place would make the control's
provenance unverifiable. Everything it validates is validated here too.

What this file adds, and why each piece is necessary
---------------------------------------------------

1. ``collect_paired_images`` is patched as well. This is the one change that
   fails SILENTLY if it is missing, so it is the reason this file exists.

   ``torch_loop._build`` line 75 calls ``collect_images``, which the original
   wrapper patches to pin the pool to the approved manifest rows. But line 80
   OVERWRITES that result in the paired branch:

       paths = config_mod.collect_images(cfg.image_root)              # patched
       if cfg.adaptation_mode == "original_transformed_consistency":
           paths = config_mod.collect_paired_images(...)              # not patched

   So in the treated arm the dataset would read whatever is on disk while the
   pool validation of lines 51-64 kept checking the manifest - decorative.

   Worse, and this is the part that corrupts rather than merely loosens: line
   280 binds the sampling plan to ``collect_images`` output, while the dataset
   iterates ``collect_paired_images`` output. If the two orders differ, the plan
   indexes different images than it believes, every recorded id belongs to a
   different frame, and the coverage accounting is wrong without raising. Hence
   ORDER PARITY is asserted here, not assumed.

2. The authorized-configuration whitelist is extended to the two arms, because
   the original does not mention ``adaptation_mode`` or
   ``tone_consistency_weight`` at all - a treated config would slip through
   unexamined.

3. ``paired_factor_gate.validate_paired_factor`` runs before training, against
   the control's EFFECTIVE resolved params. The original wrapper seals one
   configuration and never compares two.

4. The two properties that gate cannot see are checked here: it compares
   dictionaries, so it cannot notice that the arms cover different numbers of
   files, nor that an image and its transformed twin differ in size. Both were
   live failure modes on 2026-11-04.

5. The frozen tree is not touched. Its ``.py`` files are hashed at start and
   re-hashed at the end exactly as the original does, so adding a module there
   would invalidate the seal of every previous run.
"""

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

from run_overlay_ssl import (CoverageState, LocalScalarTracker, bind_sampling_ids, digest,
                             finite_gradient_norm, validate_output_dir,
                             validate_wrapper_config)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"

CONTROL_MODE = "two_view_ssl"
TREATED_MODE = "original_transformed_consistency"


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


def validate_tone_config(cfg, config):
    """Refuse any configuration outside the authorized two arms of the contrast.

    Mirrors the original whitelist value for value, then adds the tone fields it
    does not know about. Written as an explicit pair of branches rather than a
    single expression because the two arms differ in which fields MUST be empty,
    and a silent default on either side is how an inert treated arm would pass.
    """
    if (cfg.method != "simsiam_dinov2" or cfg.backbone != "dinov2_vitb14"
            or cfg.freeze_blocks != 8 or cfg.anchor_distill != 1.0
            or cfg.anchor_l2sp != 0.0001 or cfg.batch_size_per_gpu != 16
            or cfg.clinical_batch_quota not in (1, 4) or cfg.max_steps not in (40, 6000)
            or cfg.seed not in (42, 43, 44) or cfg.precision != "bf16"
            or cfg.head_warmup_steps != 32 or cfg.num_workers != 0
            or (cfg.clinical_batch_quota == 1 and config["lr_multiplier"] != 1)):
        raise ValueError("Configuration is outside the authorized isolated experiment")

    if cfg.adaptation_mode == CONTROL_MODE:
        if cfg.tone_consistency_weight != 0.0:
            raise ValueError("The control arm must carry a zero tone weight")
        if cfg.transformed_image_root or cfg.transformed_manifest_path:
            raise ValueError("The control arm must not declare transformed inputs")
    elif cfg.adaptation_mode == TREATED_MODE:
        if not cfg.tone_consistency_weight > 0.0:
            raise ValueError("The treated arm with a zero tone weight measures nothing")
        if not (cfg.transformed_image_root and cfg.transformed_manifest_path
                and cfg.transformed_data_revision and cfg.data_revision):
            raise ValueError("The treated arm must declare every transformed-manifest field")
    else:
        raise ValueError(f"Unsupported adaptation mode {cfg.adaptation_mode!r}")


def check_arm_parity(cfg, rows, evidence):
    """Check the two properties paired_factor_gate structurally cannot see.

    The gate compares two configuration dictionaries. It never counts files on
    disk and never opens an image, so neither of these would reach it:

    * the arms covering different numbers of frames - the control ran on 34,754
      while the first mirror held 33,620, and the contrast would have passed
      every check while measuring two things;
    * an image and its transformed twin differing in size, which would make the
      tone term compare tensors of different shape or, worse, silently resample.
    """
    from PIL import Image

    pairs = list(evidence.pairs)
    if len(pairs) != len(rows):
        raise ValueError(
            f"Transformed pairs ({len(pairs)}) do not cover the approved pool ({len(rows)})"
        )

    approved = {Path(row["image_path"]).resolve() for row in rows}
    paired = {original.resolve() for original, _ in pairs}
    if approved != paired:
        raise ValueError("Transformed pairs do not match the approved pool frame for frame")

    # Size parity on a deterministic sample: opening 34,754 image pairs before
    # training would cost more than it buys, and a mismatch is a property of the
    # materialization, not of individual frames.
    rng = random.Random(cfg.seed)
    for original, transformed in rng.sample(pairs, min(256, len(pairs))):
        with Image.open(original) as image:
            left = image.size
        with Image.open(transformed) as image:
            right = image.size
        if left != right:
            raise ValueError(
                f"Size mismatch between {original} {left} and {transformed} {right}"
            )


def run(config_path):
    started = time.monotonic()
    config = validate_wrapper_config(json.loads(Path(config_path).read_text()))
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
    validate_tone_config(cfg, config)

    # The single-factor comparison, against the control's EFFECTIVE params. The
    # launch JSON would check the intention; resolved-params.json is what ran.
    # paired_factor_gate lives in the repo and not in the frozen tree, so it is
    # imported from the contrast directory rather than from dermssl.
    treated_arm = cfg.adaptation_mode == TREATED_MODE
    changed_fields = None
    if treated_arm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from paired_factor_gate import validate_paired_factor
        control = json.loads(Path(config["control_resolved_params"]).read_text())
        if digest(config["control_resolved_params"]) != config["control_resolved_params_sha256"]:
            raise ValueError("Control resolved params changed since they were declared")
        effective = {**base, **{k: getattr(cfg, k) for k in base if hasattr(cfg, k)}}
        changed_fields = sorted(validate_paired_factor(control, effective))

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

    ordered = [Path(row["image_path"]) for row in rows]
    evidence = None
    if treated_arm:
        evidence = config_mod.validate_transformed_manifest(cfg)
        check_arm_parity(cfg, rows, evidence)

    (out / "launch-seal.json").write_text(json.dumps({
        "wrapper_sha256": seal,
        "wrapper_sha256_is": "sha256 of the CONFIGURATION, not of this file; the name is inherited",
        "config": config, "source_sha256": source_hashes,
        "deterministic_algorithms": True,
        "arm": TREATED_MODE if treated_arm else CONTROL_MODE,
        "paired_factor_changed_fields": changed_fields,
        "transformed_manifest_sha256": evidence.manifest_sha256 if evidence else None,
        "pool_frames": len(rows),
    }, indent=2))
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

    def paired_images(*args, **kwargs):
        """Pin the paired pool to the approved rows, IN THE ROW ORDER.

        Order is not cosmetic here. The sampling plan is bound to the patched
        ``collect_images`` output, which is ``ordered``; the dataset iterates
        what this returns. A different order means the plan indexes frames it
        did not intend, so the parity is asserted rather than trusted.
        """
        by_original = {original.resolve(): (original, transformed)
                       for original, transformed in evidence.pairs}
        pairs = []
        for path in ordered:
            key = path.resolve()
            if key not in by_original:
                raise ValueError(f"Approved frame has no transformed pair: {path}")
            pairs.append(by_original[key])
        if [original for original, _ in pairs] != ordered:
            raise ValueError("Paired pool order diverged from the sampling-plan order")
        return pairs

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
    patches = [
        patch.object(config_mod, "collect_images", return_value=ordered),
        patch.object(torch_loop.sampling_contract, "load_sampling_plan",
                     side_effect=lambda p, s, images: original_sampling(p, s, bind_sampling_ids(rows, images))),
        patch.object(config_mod, "effective_lr",
                     side_effect=lambda *a: original_lr(*a) * config["lr_multiplier"]),
        patch.object(torch_loop, "_build", side_effect=build),
        patch.object(torch_simsiam, "_build_backbone", side_effect=local_backbone),
        patch.object(torch_loop, "_save_checkpoint", side_effect=save),
        patch.object(torch_loop, "_clip_gradients",
                     side_effect=lambda *a: finite_gradient_norm(original_clip(*a))),
        patch.object(torch_loop.tracking, "build_tracker", return_value=tracker),
    ]
    if treated_arm:
        patches.append(patch.object(config_mod, "collect_paired_images", side_effect=paired_images))
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
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
                  "arm": TREATED_MODE if treated_arm else CONTROL_MODE,
                  "tone_consistency_weight": cfg.tone_consistency_weight,
                  "paired_factor_changed_fields": changed_fields,
                  "coverage": coverage.checkpoint_payload()}
        (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({key: value for key, value in result.items() if key != "coverage"}), flush=True)
    return result


if __name__ == "__main__":
    run(sys.argv[1])
