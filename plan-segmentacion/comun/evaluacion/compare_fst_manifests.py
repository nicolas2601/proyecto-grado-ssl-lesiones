"""Compare the tone and FST distributions of two FST manifests (a vs b).

Pre-registered in docs/27 ("FST con la cache sin pelo", step 1 of docs/26
§6.4): the FST manifest is rebuilt from the original cache and from the S2(a)
hair-free cache, with the SAME pseudo-masks (final ``pseudo_r2``, the ones
behind the published manifest, RUNBOOK_CLUSTER §12) and the SAME dermoscope
field (final ``cache/field.npy``), so only the pixels change. ``--images`` is
what makes the cache matter: without it the builder decodes the untouched
source files and both runs would be identical.

For every scope (``pool`` plus each dataset) and each distribution
(``tone_distribution``, ``FST_distribution``) the output has, per bin, the
counts, the proportions over that distribution's total and the difference
``b - a``; ``flagged`` carries the flagged-record counts of each manifest
(``None`` for a manifest built before the flagged block existed).

Pod commands (CPU only, no GPU; from the repository root on the pod)::

    cd ~/work/proyectonp && SAL=salidas && OUT=$SAL/v7_seg/fst_hair && mkdir -p $OUT
    # the S2(a) cache must index the same images as the masks: must print nothing
    cmp $SAL/final/cache/manifest.csv $SAL/v7_seg/s2a/cache/manifest.csv
    # (a) original cache
    python plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py \\
        --manifest $SAL/final/cache/manifest.csv \\
        --pseudo-masks $SAL/final/pseudo_r2/pseudo_masks.npy \\
        --field $SAL/final/cache/field.npy \\
        --images $SAL/final/cache/images.npy \\
        --output-json $OUT/original/fst_pseudo_manifest.json \\
        --output-csv $OUT/original/fst_pseudo_manifest.csv > $OUT/original.log 2>&1
    # (b) S2(a) hair-free cache: only --images changes
    python plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py \\
        --manifest $SAL/final/cache/manifest.csv \\
        --pseudo-masks $SAL/final/pseudo_r2/pseudo_masks.npy \\
        --field $SAL/final/cache/field.npy \\
        --images $SAL/v7_seg/s2a/cache/images.npy \\
        --output-json $OUT/s2a/fst_pseudo_manifest.json \\
        --output-csv $OUT/s2a/fst_pseudo_manifest.csv > $OUT/s2a.log 2>&1
    # sanity check: (a) must reproduce the published manifest, every diff 0
    python plan-segmentacion/comun/evaluacion/compare_fst_manifests.py \\
        --a $SAL/final/fst/fst_pseudo_manifest.json --label-a published \\
        --b $OUT/original/fst_pseudo_manifest.json --label-b original_images_npy \\
        --output-json $OUT/sanity_published_vs_original.json
    # the registered comparison
    python plan-segmentacion/comun/evaluacion/compare_fst_manifests.py \\
        --a $OUT/original/fst_pseudo_manifest.json --label-a original \\
        --b $OUT/s2a/fst_pseudo_manifest.json --label-b s2a_hair_free \\
        --output-json $OUT/comparison.json

Each manifest run is single-process CPU work over the 58 457 cached frames at
224 px: about 10 ms per frame measured locally (9.7 ms on 300 synthetic
frames), so roughly 10-20 min per run once the 8.8 GB ``images.npy`` is read
from NFS. Launch the runs detached as the runbook asks
(``setsid nohup ... < /dev/null & disown``); the comparison takes seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dermssl import run_environment  # noqa: E402
from fst_flags import TONE_LABELS  # noqa: E402

DISTRIBUTIONS = ("tone_distribution", "FST_distribution")
FLAG_COUNTS = ("n", "n_artifact_detected", "n_confidence_below_1")
POOL = "pool"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--a", type=Path, required=True, help="Reference FST manifest JSON")
    parser.add_argument("--b", type=Path, required=True, help="Compared FST manifest JSON")
    parser.add_argument("--label-a", default="a", help="Name of the reference run")
    parser.add_argument("--label-b", default="b", help="Name of the compared run")
    parser.add_argument("--output-json", type=Path, required=True, help="Comparison JSON")
    return parser.parse_args(argv)


def _ordered_bins(name: str, a: dict[str, int], b: dict[str, int]) -> list[str]:
    """Tone bins in dark/medium/light order, FST buckets in numeric order."""
    keys = set(a) | set(b)
    if name == "tone_distribution":
        return [label for label in TONE_LABELS if label in keys] + sorted(keys - set(TONE_LABELS))
    return sorted(keys, key=lambda key: (not key.isdigit(), int(key) if key.isdigit() else 0, key))


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def _bin(a_count: int, b_count: int, a_total: int, b_total: int) -> dict[str, Any]:
    a_prop, b_prop = _share(a_count, a_total), _share(b_count, b_total)
    both = a_prop is not None and b_prop is not None
    return {
        "a_count": a_count, "b_count": b_count, "a_prop": a_prop, "b_prop": b_prop,
        "diff_count": b_count - a_count, "diff_prop": b_prop - a_prop if both else None,
    }


def compare_distribution(name: str, a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    """Per-bin counts, proportions and ``b - a`` differences of one distribution."""
    a_total, b_total = sum(a.values()), sum(b.values())
    bins = {
        key: _bin(a.get(key, 0), b.get(key, 0), a_total, b_total)
        for key in _ordered_bins(name, a, b)
    }
    return {"total": {"a": a_total, "b": b_total}, "bins": bins}


def _flag_counts(summary: dict[str, Any]) -> dict[str, int] | None:
    flagged = summary.get("flagged")
    return None if flagged is None else {key: int(flagged[key]) for key in FLAG_COUNTS}


def _pool(by_dataset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Sum of every dataset summary: counts add across datasets."""
    summaries = list(by_dataset.values())
    pooled: dict[str, Any] = {"n": sum(int(item["n"]) for item in summaries)}
    for name in DISTRIBUTIONS:
        pooled[name] = dict(sum((Counter(item[name]) for item in summaries), Counter()))
    flags = [_flag_counts(item) for item in summaries]
    if flags and all(flag is not None for flag in flags):
        pooled["flagged"] = {key: sum(flag[key] for flag in flags) for key in FLAG_COUNTS}
    return pooled


def _scope(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": {"a": int(a["n"]), "b": int(b["n"])}}
    for name in DISTRIBUTIONS:
        result[name] = compare_distribution(name, a[name], b[name])
    result["flagged"] = {"a": _flag_counts(a), "b": _flag_counts(b)}
    return result


def _by_dataset(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return manifest["summary"]["by_dataset"]


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Scope-by-scope comparison of two manifests covering the same datasets."""
    a_sets, b_sets = _by_dataset(a), _by_dataset(b)
    if set(a_sets) != set(b_sets):
        raise ValueError(f"datasets differ: {sorted(set(a_sets) ^ set(b_sets))}")
    scopes = {POOL: _scope(_pool(a_sets), _pool(b_sets))}
    scopes.update({name: _scope(a_sets[name], b_sets[name]) for name in sorted(a_sets)})
    return {"diff": "b - a", "scopes": scopes}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = {
        "labels": {"a": args.label_a, "b": args.label_b},
        "sources": {"a": str(args.a), "b": str(args.b)},
        **compare(_load(args.a), _load(args.b)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    run_environment.write_sidecar(args.output_json, args)
    print(json.dumps(result["scopes"][POOL], indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
