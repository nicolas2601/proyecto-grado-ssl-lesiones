"""The three fixed configurations of the E1 bank and how their cells are named (docs/33 sections 2 and 3).

Nothing here is selected on validation: each config is the one v7.1 registered.

* ``B71``  = ``B_s42_a1.0@224|cls4|balanced|tta``
* ``A71``  = ``A_soup_a1.0@224|cls_attn|la0.5|tta``
* ``PUBLIC_B71`` / ``PUBLIC_A71`` = public DINOv2 under the protocol of B71 / A71
  (``v71_protocol.with_weights``, how v7.1 matched public to a chosen config).

B71 and A71 exist twice, ``original`` (SSL pool contained the val images, so the row is marked
transductive) and ``clean`` (pool excluded the val ids, docs/33 section 2).
"""

from __future__ import annotations

from dermssl import v71_protocol as vp

RESOLUTION = vp.SELECTION_RESOLUTION
B71 = vp.config_id("B_s42_a1.0", RESOLUTION, "cls4", "balanced", True)
A71 = vp.config_id("A_soup_a1.0", RESOLUTION, "cls_attn", "la0.5", True)
CONFIGS = {
    "B71": B71,
    "A71": A71,
    "PUBLIC_B71": vp.with_weights(B71, vp.PUBLIC),
    "PUBLIC_A71": vp.with_weights(A71, vp.PUBLIC),
}
CLI_CONFIGS = {"B71": ("B71",), "A71": ("A71",), "PUBLIC": ("PUBLIC_B71", "PUBLIC_A71")}
PARTS_OF = {"cls4": ("cls4",), "cls_attn": ("cls", "attn")}
VARIANTS = ("original", "clean")
# run.json exclude_ids_sha256 of the weights: v7 exclusion (docs/33 section 2) and the clean one.
POOL_EXCLUDE_SHA256 = {
    "original": "95d21013a42a5c99a6024ef6c8ae73aada8bdc1193ae512665035bb59b737d87",
    "clean": "3b7e883392ba5ae0cb290166fa2d4e2dfbb325d4bce0bca3e96562f8c39dfc1d",
}


def is_public(config: str) -> bool:
    return config.startswith("PUBLIC")


def variant_of(pool_excludes_val: bool) -> str:
    return "clean" if pool_excludes_val else "original"


def cell_name(config: str, pool_excludes_val: bool) -> str:
    """``PUBLIC_B71`` (no SSL pool) or ``B71__clean`` / ``B71__original``."""
    return config if is_public(config) else f"{config}__{variant_of(pool_excludes_val)}"


def transductive(config: str, pool_excludes_val: bool) -> bool:
    """True when the SSL pool saw the val images; public DINOv2 never saw this project's pool."""
    return not is_public(config) and not pool_excludes_val


def parse(config: str) -> dict:
    return vp.parse_config_id(CONFIGS[config])


def parts_needed(config: str) -> tuple[str, ...]:
    return PARTS_OF[parse(config)["feature"]]


def contrast_pairs() -> list[tuple[str, str]]:
    """``(arm cell, control cell)``: each config against public and the other config, clean against original."""
    pairs = []
    for variant in VARIANTS:
        b, a = f"B71__{variant}", f"A71__{variant}"
        pairs += [(b, "PUBLIC_B71"), (a, "PUBLIC_A71"), (b, a)]
    return pairs + [("B71__clean", "B71__original"), ("A71__clean", "A71__original")]
