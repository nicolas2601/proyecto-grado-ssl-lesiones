"""Where the finals live and which SSL pool they saw (docs/33 section 2).

``original`` finals are ``salidas/v7/final_{A,B}`` (seed 42) and ``final_{A,B}_s43``; ``clean`` finals are
``salidas/bench/v7clean/final_{A,B}_s{42,43}``. The ``transductivo`` label is only trusted when each
final's ``run.json`` records the exclusion file the label implies.
"""

from __future__ import annotations

from dermssl.isic_eval import protocol as pr

BASE_SEED = 42


class PoolLabelError(RuntimeError):
    """The weights' recorded SSL exclusion does not match the declared pool variant."""


def final_dir_name(arm: str, seed: int, pool_excludes_val: bool) -> str:
    if pool_excludes_val or seed != BASE_SEED:
        return f"final_{arm}_s{seed}"
    return f"final_{arm}"


def check_pool_label(manifests: dict, pool_excludes_val: bool, allow_unverified: bool = False) -> dict:
    """Refuse a mislabelled run: every ``run.json`` ``exclude_ids_sha256`` must be the declared variant's."""
    variant = pr.variant_of(pool_excludes_val)
    expected = pr.POOL_EXCLUDE_SHA256[variant]
    recorded = {name: run.get("exclude_ids_sha256") for name, run in manifests.items()}
    wrong = {name: sha for name, sha in recorded.items() if sha != expected}
    if wrong and not (allow_unverified and all(sha is None for sha in wrong.values())):
        raise PoolLabelError(f"--pool-excludes-val={pool_excludes_val} expects exclude_ids_sha256 "
                             f"{expected[:12]} ({variant}); got "
                             f"{ {k: (v or 'none')[:12] for k, v in wrong.items()} }")
    return {"variant": variant, "exclude_ids_sha256": recorded, "verified": not wrong}
