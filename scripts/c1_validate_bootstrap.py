"""Validate the paired bootstrap against a contrast the v6 already published.

Why
---
``resultados/clasificacion/bootstrap.json`` has no generator in the repo, so
the new implementation has nothing to be checked against except its own
tests. Those prove it is self-consistent, not that it measures what the v6
measured.

The point estimate is the handle: it does not depend on resampling at all, so
a correct reimplementation must reproduce it EXACTLY from the same features.
If it does not, the two are not measuring the same thing and no contrast C1
produces would be comparable to the published table.

The interval is different -- 1000 resamples of a random process reproduce
only up to Monte Carlo noise -- so it is reported with its deviation rather
than asserted.

Usage
-----
    python scripts/c1_validate_bootstrap.py \\
        --cache-dir ~/work/proyectonp/salidas/tone/bank/cache \\
        --published resultados/clasificacion/bootstrap.json \\
        --arm ssl_proposed_vits14 --control ssl_plain_vits14 --metric macro_f1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import probes  # noqa: E402
from dermssl.feature_cache import FeatureCache  # noqa: E402

#: Absolute tolerance on the point estimate. Not a judgement call: two runs of
#: the same arithmetic on the same float32 features differ only by the order
#: of operations inside BLAS, which is well under this.
POINT_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--metric", default="macro_f1")
    parser.add_argument("--probe", default="logreg")
    return parser.parse_args()


def load_split(cache: FeatureCache, backbone: str, split: str):
    """Find this backbone's entry for a split by its filename fingerprint."""
    hits = sorted(cache.root.glob(f"v1__{backbone}__{split}__r*__eval__*.features.npy"))
    if len(hits) != 1:
        raise SystemExit(
            f"ABORT: esperaba 1 cache de {backbone}/{split}, encontre {len(hits)}"
        )
    key = hits[0].name[: -len(".features.npy")]
    entry = cache.load(key)
    if entry is None:
        raise SystemExit(f"ABORT: el cache {key} no se pudo cargar")
    return entry


def probe_proba(cache: FeatureCache, backbone: str, probe: str) -> tuple:
    train = load_split(cache, backbone, "train")
    val = load_split(cache, backbone, "val")
    n_classes = int(max(train.labels.max(), val.labels.max())) + 1
    proba = probes.run_probe(
        probe, train.features, train.labels, val.features, n_classes
    )
    return np.asarray(proba), np.asarray(val.labels), n_classes


def main() -> int:
    args = parse_args()
    published = json.loads(args.published.expanduser().read_text(encoding="utf-8"))
    wanted = [
        r
        for r in published["results"]
        if r["arm"] == args.arm and r["control"] == args.control and r["metric"] == args.metric
    ]
    if len(wanted) != 1:
        raise SystemExit(
            f"ABORT: encontre {len(wanted)} contrastes para "
            f"{args.arm} vs {args.control} / {args.metric}"
        )
    reference = wanted[0]

    cache = FeatureCache(args.cache_dir.expanduser())
    arm_proba, y_true, _ = probe_proba(cache, args.arm, args.probe)
    control_proba, y_control, _ = probe_proba(cache, args.control, args.probe)

    if not np.array_equal(y_true, y_control):
        raise SystemExit(
            "ABORT: los dos brazos tienen etiquetas de validacion distintas; no son el "
            "mismo split y el contraste pareado no tendria sentido"
        )

    record = pb.contrast_record(
        arm=args.arm,
        control=args.control,
        metric=args.metric,
        probe=args.probe,
        fraction=published["fraction"],
        seed=published["seed_used"],
        y_true=y_true,
        arm_proba=arm_proba,
        control_proba=control_proba,
        n_resamples=published["n_resamples"],
        rng_seed=published["rng_seed"],
    )

    print(f"contraste: {args.arm} vs {args.control} / {args.metric}")
    print(f"{'':16} {'publicado':>16} {'regenerado':>16} {'|delta|':>12}")
    ok = True
    for field in ("point_arm", "point_control", "point_delta"):
        difference = abs(record[field] - reference[field])
        flag = "" if difference <= POINT_TOLERANCE else "  <-- NO COINCIDE"
        ok = ok and difference <= POINT_TOLERANCE
        print(f"{field:16} {reference[field]:16.12f} {record[field]:16.12f} "
              f"{difference:12.3e}{flag}")

    # The interval is a random quantity; reported with its deviation, not gated.
    for field in ("ci95_low", "ci95_high"):
        difference = abs(record[field] - reference[field])
        print(f"{field:16} {reference[field]:16.12f} {record[field]:16.12f} "
              f"{difference:12.3e}  (ruido Monte Carlo, informativo)")

    print(f"\ndirection publicada: {reference['direction']}  |  regenerada: {record['direction']}")
    if record["direction"] != reference["direction"]:
        print("  <-- NO COINCIDE: el veredicto cambia de lado")
        ok = False

    if not ok:
        print(
            "\n=== EL PUNTO ESTIMADO NO COINCIDE CON LA v6 ===\n"
            "El punto no depende del remuestreo, asi que esto no es ruido: significa que "
            "no estamos midiendo lo mismo que la v6 y ningun contraste de C1 seria "
            "comparable con la tabla publicada."
        )
        return 1

    print("\n=== VALIDADO: el punto estimado reproduce la v6 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
