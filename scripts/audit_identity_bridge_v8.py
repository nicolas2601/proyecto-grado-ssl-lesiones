"""Cross the v8 training pools against the evaluation split inside one identifier namespace.

Three namespaces are in play and they are not interchangeable:

    IL_*  / IP_*            ISIC Archive lesion / patient ids
    HAM_* / BCN_* / MSK4_*  per-contributor lesion ids in the local ISIC 2019 CSV

Comparing a string from one against a string from another returns zero by
construction. This script therefore does two separate things and never mixes them:

1. **Single-namespace cross.** Uses the archive snapshot for BOTH sides, so pool
   and evaluation identifiers are comparable, and reports collisions together
   with the coverage that bounds them.

2. **Bridge construction.** Images that carry an archive id *and* a contributor
   id are bridge observations: they assert ``IL_x == HAM_y``. The script measures
   whether the two groupings agree on the overlap, which is what decides if the
   contributor ids can be used to extend coverage or must be discarded.

A collision count is never reported without its coverage. An identifier that is
absent is UNKNOWN, never independent.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MISSING = {"", "nan", "none", "null", "unknown", "na", "n/a"}


def is_set(value: str | None) -> bool:
    return (value or "").strip().casefold() not in MISSING


def numeric_id(image_id: str) -> int:
    return int(re.sub(r"[^0-9]", "", image_id))


def load_pool(path: Path) -> dict[str, str]:
    pool = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_id, _, dataset = line.partition("\t")
        pool[image_id] = dataset
    return pool


def load_eval(path: Path) -> list[str]:
    return [json.loads(line)["image_id"] for line in path.open(encoding="utf-8")]


def load_snapshot(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def load_contributor_lesions(path: Path) -> dict[str, str]:
    lesions = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = (row.get("image") or "").strip()
            if image_id and is_set(row.get("lesion_id")):
                lesions[image_id] = row["lesion_id"].strip()
    return lesions


def cross(
    pool: dict[str, str],
    eval_ids: list[str],
    identities: dict[str, str | None],
    label: str,
) -> dict:
    """Collisions of one identifier field between a pool and the evaluation split."""
    pool_ids = {i: identities[i] for i in pool if identities.get(i)}
    eval_ids_known = {i: identities[i] for i in eval_ids if identities.get(i)}
    shared = set(pool_ids.values()) & set(eval_ids_known.values())

    affected_eval = sorted(i for i, v in eval_ids_known.items() if v in shared)
    affected_pool = sorted(i for i, v in pool_ids.items() if v in shared)

    return {
        "field": label,
        "pool_size": len(pool),
        "pool_with_id": len(pool_ids),
        "pool_coverage": len(pool_ids) / len(pool),
        "eval_size": len(eval_ids),
        "eval_with_id": len(eval_ids_known),
        "eval_coverage": len(eval_ids_known) / len(eval_ids),
        "shared_identities": len(shared),
        "eval_images_affected": len(affected_eval),
        "pool_images_affected": len(affected_pool),
        "eval_examples": affected_eval[:15],
        "measurable": bool(pool_ids) and bool(eval_ids_known),
    }


def build_bridge(
    archive_lesion: dict[str, str | None], contributor_lesion: dict[str, str]
) -> dict:
    """Do archive lesion ids and contributor lesion ids induce the same grouping?"""
    both = [i for i in contributor_lesion if archive_lesion.get(i)]
    mapping: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for image_id in both:
        archive = archive_lesion[image_id]
        contributor = contributor_lesion[image_id]
        mapping[archive].add(contributor)
        reverse[contributor].add(archive)

    ambiguous_forward = {k: sorted(v) for k, v in mapping.items() if len(v) > 1}
    ambiguous_reverse = {k: sorted(v) for k, v in reverse.items() if len(v) > 1}

    return {
        "bridge_observations": len(both),
        "archive_lesions_bridged": len(mapping),
        "contributor_lesions_bridged": len(reverse),
        "archive_id_mapping_to_multiple_contributor_ids": len(ambiguous_forward),
        "contributor_id_mapping_to_multiple_archive_ids": len(ambiguous_reverse),
        "contributor_prefixes": dict(
            Counter(v.split("_")[0] for v in contributor_lesion.values()).most_common()
        ),
        "one_to_one": not ambiguous_forward and not ambiguous_reverse,
        "examples_ambiguous_reverse": dict(list(ambiguous_reverse.items())[:5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--eval-cases", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--contributor-lesions", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    snapshot = load_snapshot(args.snapshot)
    archive_patient = {k: v.get("patient_id") for k, v in snapshot.items()}
    archive_lesion = {k: v.get("lesion_id") for k, v in snapshot.items()}
    eval_ids = load_eval(args.eval_cases)

    result: dict = {
        "snapshot_records": len(snapshot),
        "snapshot_complete": json.loads(args.snapshot.read_text(encoding="utf-8")).get(
            "complete"
        ),
        "crosses": [],
    }

    print(f"snapshot del archivo ISIC: {len(snapshot)} registros")
    print(f"snapshot completo: {result['snapshot_complete']}\n")

    for pool_path in args.pool:
        pool = load_pool(pool_path)
        name = pool_path.stem
        missing_from_snapshot = sum(1 for i in pool if i not in snapshot)
        print(f"=== {name}: {len(pool)} imagenes, {missing_from_snapshot} fuera del snapshot ===")
        for field, identities in (("lesion_id", archive_lesion), ("patient_id", archive_patient)):
            block = cross(pool, eval_ids, identities, field)
            block["pool"] = name
            block["pool_images_missing_from_snapshot"] = missing_from_snapshot
            result["crosses"].append(block)
            state = "MEDIBLE" if block["measurable"] else "NO MEDIBLE"
            print(
                f"  {field:<10} [{state}] cobertura pool {block['pool_coverage']:.1%} "
                f"/ eval {block['eval_coverage']:.1%}"
            )
            print(
                f"    identidades compartidas: {block['shared_identities']} | "
                f"imagenes del eval afectadas: {block['eval_images_affected']} "
                f"({block['eval_images_affected'] / block['eval_size']:.2%})"
            )
            if block["eval_examples"]:
                print(f"    ejemplos: {', '.join(block['eval_examples'][:6])}")
        print()

    if args.contributor_lesions:
        contributor = load_contributor_lesions(args.contributor_lesions)
        bridge = build_bridge(archive_lesion, contributor)
        result["bridge"] = bridge
        print("=== PUENTE entre lesion_id del archivo y de los contribuyentes ===")
        print(f"  observaciones de puente: {bridge['bridge_observations']}")
        print(f"  prefijos de contribuyente: {bridge['contributor_prefixes']}")
        print(
            f"  un IL_ -> varios contribuyentes: "
            f"{bridge['archive_id_mapping_to_multiple_contributor_ids']}"
        )
        print(
            f"  un contribuyente -> varios IL_: "
            f"{bridge['contributor_id_mapping_to_multiple_archive_ids']}"
        )
        print(f"  biyectivo: {bridge['one_to_one']}")

    # Structural observation: pool and evaluation ids interleave in the same range,
    # which is why same-lesion leakage is plausible rather than hypothetical.
    eval_numeric = sorted(numeric_id(i) for i in eval_ids)
    eval_set = set(eval_numeric)
    print(f"\n=== rango de id: eval {eval_numeric[0]}-{eval_numeric[-1]} ===")
    interleave = {}
    for pool_path in args.pool:
        pool = load_pool(pool_path)
        in_range = [i for i in pool if eval_numeric[0] <= numeric_id(i) <= eval_numeric[-1]]
        adjacent = [
            i for i in in_range if any(numeric_id(i) + d in eval_set for d in (-2, -1, 1, 2))
        ]
        interleave[pool_path.stem] = {
            "in_eval_id_range": len(in_range),
            "adjacent_within_2": len(adjacent),
            "examples": sorted(adjacent)[:8],
        }
        print(
            f"  {pool_path.stem}: {len(in_range)} en el rango del eval, "
            f"{len(adjacent)} a distancia <= 2 de un id del eval"
        )
    result["id_range_interleaving"] = interleave

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
