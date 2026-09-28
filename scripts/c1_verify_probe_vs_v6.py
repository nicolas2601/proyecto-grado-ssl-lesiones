"""Exact check: the runner's own probe+metrics on the v6 CACHE features.

Separates code from precision. C1 extracts in fp32 and the v6 cache was
extracted under autocast, so C1's condition (0) cannot match the v6 to the
decimal. But if the runner's evaluate_cell -- the SAME function the grid uses
-- is run on the cache features themselves, every metric must come out
identical to the published v6 values. That proves the probe and the metrics
are the v6's; any remaining gap in the grid is then precision alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import c1_run_grid as grid  # noqa: E402
from dermssl.feature_cache import FeatureCache  # noqa: E402

TOLERANCE = 1e-9
BACKBONES = ("dinov2_public_vitb14", "ssl_proposed_vitb14")
REFERENCE = {"dinov2_public_vitb14": "point_control", "ssl_proposed_vitb14": "point_arm"}


def main() -> int:
    cache_dir = Path(sys.argv[1]).expanduser()
    published = json.loads((REPO / "resultados/clasificacion/bootstrap.json").read_text())
    rows = {
        r["metric"]: r for r in published["results"]
        if r["arm"] == "ssl_proposed_vitb14" and r["control"] == "dinov2_public_vitb14"
    }
    cache = FeatureCache(cache_dir)
    ok = True
    for backbone in BACKBONES:
        split = {}
        for name in ("train", "val"):
            hit = sorted(cache_dir.glob(f"v1__{backbone}__{name}__r*__eval__*.features.npy"))[0]
            split[name] = cache.load(hit.name[: -len(".features.npy")])
        n_classes = int(max(split["train"].labels.max(), split["val"].labels.max())) + 1
        _, metrics = grid.evaluate_cell(
            split["train"].features, split["train"].labels,
            split["val"].features, split["val"].labels, "logreg", n_classes,
        )
        print(f"\n{backbone}")
        for metric in ("macro_f1", "balanced_accuracy", "auc", "ece"):
            want = rows[metric][REFERENCE[backbone]]
            have = float(metrics[metric])
            diff = abs(have - want)
            flag = "" if diff <= TOLERANCE else "  <-- NO COINCIDE"
            ok = ok and diff <= TOLERANCE
            print(f"  {metric:18} publicado {want:.12f}  runner {have:.12f}  |d| {diff:.1e}{flag}")
    print("\n=== EXACTO: probe y metricas del runner son los de la v6 ===" if ok
          else "\n=== NO COINCIDE: el runner no mide lo mismo que la v6 ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
