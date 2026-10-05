"""Which B* seed the E3 B71 feature comes from (post-hoc robustness run, docs/33 errata).

The registered E3 runs on seed 42. The robustness run recomputes the SAME pinned
comparison with the seed 43 and seed 44 finals, so everything that depends on the seed
lives here: the weights folder (the v7 layout), the feature key, the protocol string, a
typed refusal when the weights are absent, and the fingerprint fields that keep cells of
different seeds from ever being mixed. Seed 42 adds no new fingerprint key, so its cells
and sidecars stay byte-identical to the registered run.
"""

from __future__ import annotations

from pathlib import Path

from dermssl import e3_store as st
from dermssl import run_provenance as rp
from dermssl import v7_seeds as seeds
from dermssl import v71_protocol as vp

BASE_SEED = seeds.BASE_SEED
SEEDS = seeds.REGISTERED_SEEDS
WEIGHTS_FILE = "simsiam_dinov2_backbone_final.pt"  # the bank's name; a test pins the equality
ARM, ALPHA, RESOLUTION, PART = "B", 1.0, 224, "cls4"


class UnknownSeedError(st.E3Error):
    """The seed is not one of the registered v7 seeds."""


class MissingWeightsError(st.E3Error):
    """The B* final of the requested seed is not where the v7 layout puts it."""


def _check(seed: int) -> int:
    if seed not in SEEDS:
        raise UnknownSeedError(f"semilla {seed!r} no registrada; se esperaba una de {SEEDS}")
    return seed


def weights_dir_name(seed: int) -> str:
    """``final_B`` for the base seed, ``final_B_s<seed>`` otherwise."""
    return seeds.final_dir_name(ARM, _check(seed))


def weights_name(seed: int) -> str:
    return f"{ARM}_s{_check(seed)}_a{ALPHA}"


def feature_key(seed: int) -> str:
    return f"{weights_name(seed)}@{RESOLUTION}|{PART}"


def protocol(seed: int) -> str:
    return vp.config_id(weights_name(seed), RESOLUTION, PART, "balanced", True)


def weights_path(weights_dir, seed: int) -> Path:
    """The final's weights file, or a :class:`MissingWeightsError` wrapping the OS error."""
    path = Path(weights_dir) / weights_dir_name(seed) / WEIGHTS_FILE
    try:
        path.open("rb").close()
    except OSError as exc:
        raise MissingWeightsError(f"sin pesos de B* semilla {seed} en {path}: {exc}") from exc
    return path


def fingerprint(weights_dir, seed: int) -> dict:
    """Protocol and weights sha256 (plus the seed itself when it is not the base one)."""
    path = weights_path(weights_dir, seed)
    fields = {"protocol": protocol(seed), "weights_sha256": rp.file_sha256(path)}
    return fields if seed == BASE_SEED else {**fields, "b71_seed": seed}
