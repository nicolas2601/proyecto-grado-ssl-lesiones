"""Config validation, clinical-quota sampler and L2-SP penalty for the configurable MoCo-v3-style arm (v4).

Defaults reproduce moco_v3_20261029 exactly. Tensor helpers (loss, EMA, teacher, heads, view transform) are
reused from the v3 helpers module, which is never edited.
"""

import random

import torch

from moco_v3_20261029_helpers import (  # noqa: F401  (re-exported for the v4 runtime)
    ALLOWED_ROLES,
    TEMPERATURE,
    apply_freeze_plan,
    build_heads,
    build_teacher,
    check_manifest_rows,
    cosine_anchor_loss,
    ema_update,
    make_view_transform,
    moco_symmetric_loss,
)

ALLOWED_SEEDS = frozenset({42, 43, 44})
MOMENTUM_RANGE = (0.9, 0.9999)
FREEZE_RANGE = (0, 11)
SOURCE_ROLE = "source_fit"
CLINICAL_ROLE = "pad_adapt"
REQUIRED_BASE = frozenset({"initial_weights", "output_dir", "max_steps", "seed"})
REQUIRED_TOP = frozenset({"manifest", "manifest_sha256", "manifest_roles", "batch_size", "base_config"})
OPTION_DEFAULTS = {
    "teacher_momentum": 0.99,
    "temperature": TEMPERATURE,
    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "weight_decay": 0.05,
    "freeze_blocks": 8,
    "cosine_anchor": False,
    "l2sp": 0.0,
    "clinical_quota": None,
    "checkpoint_steps": [],
}
HEX_DIGITS = frozenset("0123456789abcdef")
SHA256_LENGTH = 64


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number(options, key, low=None, high=None, strict_low=False):
    value = options[key]
    if not _is_number(value):
        raise ValueError(f"{key} must be a number")
    if low is not None and (value <= low if strict_low else value < low):
        raise ValueError(f"{key} out of range: {value}")
    if high is not None and value > high:
        raise ValueError(f"{key} out of range: {value}")


def _check_checkpoint_steps(steps, max_steps):
    if not isinstance(steps, list) or not all(_is_int(s) and 0 < s < max_steps for s in steps):
        raise ValueError("checkpoint_steps must be a list of integers in (0, max_steps)")
    if any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("checkpoint_steps must be strictly ascending")


def _check_options(options, batch_size):
    _check_number(options, "teacher_momentum", *MOMENTUM_RANGE)
    _check_number(options, "temperature", 0, strict_low=True)
    _check_number(options, "backbone_lr", 0, strict_low=True)
    _check_number(options, "head_lr", 0, strict_low=True)
    _check_number(options, "weight_decay", 0)
    _check_number(options, "l2sp", 0)
    if not _is_int(options["freeze_blocks"]) or not FREEZE_RANGE[0] <= options["freeze_blocks"] <= FREEZE_RANGE[1]:
        raise ValueError("freeze_blocks must be an integer in 0..11")
    if not isinstance(options["cosine_anchor"], bool):
        raise ValueError("cosine_anchor must be a boolean")
    quota = options["clinical_quota"]
    if quota is not None and not (_is_int(quota) and 1 <= quota <= batch_size):
        raise ValueError("clinical_quota must be null or an integer in 1..batch_size")
    _check_checkpoint_steps(options["checkpoint_steps"], options["max_steps"])


def _check_top_level(config):
    if not REQUIRED_TOP <= config.keys():
        raise ValueError(f"missing config fields: {sorted(REQUIRED_TOP - config.keys())}")
    if not _is_int(config["batch_size"]) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if set(config["manifest_roles"]) != ALLOWED_ROLES:
        raise ValueError("manifest_roles must declare source_fit and pad_adapt only")
    sha = config["manifest_sha256"]
    if not isinstance(sha, str) or len(sha) != SHA256_LENGTH or not set(sha) <= HEX_DIGITS:
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")


def validate_config(config):
    """Returns (base_config merged with option defaults, batch_size); rejects unknown or invalid fields."""
    _check_top_level(config)
    base = config["base_config"]
    if not isinstance(base, dict) or not REQUIRED_BASE <= base.keys():
        raise ValueError("base_config requires initial_weights, output_dir, max_steps, and seed")
    unknown = base.keys() - REQUIRED_BASE - OPTION_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown base_config keys: {sorted(unknown)}")
    if not _is_int(base["seed"]) or base["seed"] not in ALLOWED_SEEDS:
        raise ValueError("seed must be one of 42, 43, or 44")
    if not _is_int(base["max_steps"]) or base["max_steps"] <= 0:
        raise ValueError("max_steps must be a positive integer")
    merged = {**{k: (list(v) if isinstance(v, list) else v) for k, v in OPTION_DEFAULTS.items()}, **base}
    _check_options(merged, config["batch_size"])
    return merged, config["batch_size"]


def make_batch_sampler(rows, batch_size, quota):
    """Returns a zero-arg callable yielding batch rows; quota None keeps v3's random.sample(rows, batch_size)."""
    if quota is None:
        return lambda: random.sample(rows, batch_size)
    source = [row for row in rows if row["role"] == SOURCE_ROLE]
    clinical = [row for row in rows if row["role"] == CLINICAL_ROLE]
    need_source = batch_size - quota
    if len(source) < need_source or len(clinical) < quota:
        raise ValueError(f"clinical_quota={quota} needs {need_source} {SOURCE_ROLE} and {quota} {CLINICAL_ROLE} "
                         f"rows; manifest has {len(source)} and {len(clinical)}")
    return lambda: random.sample(source, need_source) + random.sample(clinical, quota)


def snapshot_trainable(model):
    """Frozen copy of the currently trainable parameters (the public initial weights for L2-SP)."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def l2sp_penalty(model, reference, strength):
    """strength * sum of squared deviations of trainable parameters from the frozen reference."""
    total = sum(((p.float() - reference[name].float()) ** 2).sum()
                for name, p in model.named_parameters() if p.requires_grad)
    return strength * (total if torch.is_tensor(total) else torch.zeros(()))
