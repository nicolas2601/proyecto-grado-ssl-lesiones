"""Measure patient and lesion leakage between the v8 training pool and the ISIC 2018 evaluation split.

``patient_independence_accredited`` has been ``false`` across runs 019-025 because
the available evidence has not accredited patient-level independence. An earlier attempt used the IMA++
metadata, which only covers 15% of the evaluation split and 0.47% of the training
pool, so the result was "3 collisions found in the sliver we can see" -- not a
bound.

This audit instead uses the ISIC archive metadata that already sits on the
cluster, which carries ``patient_id`` for both pool datasets and ``lesion_id``
for ISIC 2019. It reports coverage first and collisions second, because a
collision count without its coverage denominator is not interpretable.

Read-only. Computes nothing that depends on a model or a run.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path

CLUSTER_ROOT = Path("/home/jovyan/work")
DEFAULT_POOL_MANIFEST = (
    CLUSTER_ROOT
    / "dermlip_slic_v8/preflight/shared-segmentation-pools-v2-20261022-001/selection.manifest.json"
)
DEFAULT_SOURCES = {
    "isic2019_patient": CLUSTER_ROOT / "datos/isic2019/train-metadata.csv",
    "isic2020_patient": CLUSTER_ROOT / "datos/isic2020/train-metadata.csv",
    "isic2019_lesion": CLUSTER_ROOT / "datos/isic2019_fullres/ISIC_2019_Training_Metadata.csv",
}

MISSING = {"", "nan", "none", "null", "unknown", "na", "n/a"}

# A column of per-image placeholders is worse than a missing column: it makes a
# collision search return zero by construction and reads as a clean bill of
# health. The local derived ISIC 2019 CSV uses patient_id as dummy_0, dummy_1, ... one per image, so
# this guard is not hypothetical.
PLACEHOLDER_PREFIXES = ("dummy", "placeholder", "unknown", "na_", "none_")


def is_set(value: str | None) -> bool:
    return (value or "").strip().casefold() not in MISSING


def classify_identifier(values: list[str], name: str) -> dict:
    """Decide whether an identifier column can support a grouping claim at all."""
    distinct = len(set(values))
    ratio = (distinct / len(values)) if values else 0.0
    looks_placeholder = bool(values) and sum(
        1 for v in values if v.lower().startswith(PLACEHOLDER_PREFIXES)
    ) / len(values) > 0.5
    degenerate = not values or looks_placeholder
    return {
        "name": name,
        "rows": len(values),
        "distinct": distinct,
        "distinct_ratio": ratio,
        "looks_placeholder": looks_placeholder,
        "usable_for_grouping": not degenerate,
        "reason": (
            "per-image placeholder or index; a zero-collision result would be "
            "vacuous"
            if degenerate
            else "non-placeholder candidates; namespace provenance still requires verification"
        ),
    }


def load_identity_maps(
    sources: dict[str, Path],
) -> tuple[dict[str, str], dict[str, str], list[dict]]:
    """Return image_id -> patient_id, image_id -> lesion_id, and a verdict per source.

    Identifier namespaces are NOT merged blindly: each source is classified on its
    own, because one usable column does not redeem a placeholder one.
    """
    patient: dict[str, str] = {}
    lesion: dict[str, str] = {}
    diagnostics: list[dict] = []

    for key, id_column, value_column in (
        ("isic2019_patient", "isic_id", "patient_id"),
        ("isic2020_patient", "isic_id", "patient_id"),
        ("isic2019_lesion", "image", "lesion_id"),
    ):
        values: list[str] = []
        target = lesion if value_column == "lesion_id" else patient
        with sources[key].open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_id = (row.get(id_column) or "").strip()
                if image_id and is_set(row.get(value_column)):
                    value = row[value_column].strip()
                    values.append(value)
                    if value.casefold().startswith(PLACEHOLDER_PREFIXES):
                        continue
                    # Only compare identities within the same documented source.
                    qualified = f"{key}:{value}"
                    if image_id in target and target[image_id] != qualified:
                        raise ValueError(f"Conflicting identity assignments for image {image_id}")
                    target[image_id] = qualified
        diagnostics.append(classify_identifier(values, f"{key}.{value_column}"))

    return patient, lesion, diagnostics


def load_pool(manifest_path: Path) -> dict[str, str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["image_id"]: record["dataset"] for record in data["records"]}


def load_eval(cases_path: Path) -> list[str]:
    ids = []
    with cases_path.open(encoding="utf-8") as handle:
        for line in handle:
            ids.append(json.loads(line)["image_id"])
    return ids


def summarize(
    pool: dict[str, str],
    eval_ids: list[str],
    patient: dict[str, str],
    lesion: dict[str, str],
) -> dict:
    eval_set = set(eval_ids)
    pool_set = set(pool)

    pool_patient = {i: patient[i] for i in pool_set if i in patient}
    pool_lesion = {i: lesion[i] for i in pool_set if i in lesion}
    eval_patient = {i: patient[i] for i in eval_set if i in patient}
    eval_lesion = {i: lesion[i] for i in eval_set if i in lesion}

    shared_patients = set(pool_patient.values()) & set(eval_patient.values())
    shared_lesions = set(pool_lesion.values()) & set(eval_lesion.values())

    by_patient = defaultdict(list)
    for image_id, pid in pool_patient.items():
        if pid in shared_patients:
            by_patient[pid].append(image_id)
    by_lesion = defaultdict(list)
    for image_id, lid in pool_lesion.items():
        if lid in shared_lesions:
            by_lesion[lid].append(image_id)

    eval_hit_patient = sorted(i for i, pid in eval_patient.items() if pid in shared_patients)
    eval_hit_lesion = sorted(i for i, lid in eval_lesion.items() if lid in shared_lesions)

    if not pool_set or not eval_set:
        raise ValueError("Pool and evaluation cohort must both be non-empty")

    def scope_status(left, right):
        shared_namespaces = {v.partition(":")[0] for v in left.values()} & {
            v.partition(":")[0] for v in right.values()
        }
        if not left or not right or not shared_namespaces:
            return "not_measurable"
        return "partial"  # Available metadata alone never accredits independence.

    return {
        "patient_independence_accredited": False,
        "patient_comparison_status": scope_status(pool_patient, eval_patient),
        "lesion_comparison_status": scope_status(pool_lesion, eval_lesion),
        "identity_namespace_policy": "source_scoped_no_unverified_cross_source_linkage",
        "pool_size": len(pool_set),
        "eval_size": len(eval_set),
        "direct_image_id_overlap": len(pool_set & eval_set),
        "coverage": {
            "pool_with_patient_id": len(pool_patient),
            "pool_patient_coverage": len(pool_patient) / len(pool_set),
            "pool_with_lesion_id": len(pool_lesion),
            "pool_lesion_coverage": len(pool_lesion) / len(pool_set),
            "eval_with_patient_id": len(eval_patient),
            "eval_patient_coverage": len(eval_patient) / len(eval_set),
            "eval_with_lesion_id": len(eval_lesion),
            "eval_lesion_coverage": len(eval_lesion) / len(eval_set),
        },
        "patient_leakage": {
            "shared_patients": len(shared_patients),
            "pool_images_involved": sum(len(v) for v in by_patient.values()),
            "eval_images_involved": len(eval_hit_patient),
            "eval_images_involved_frac": len(eval_hit_patient) / len(eval_set),
            "eval_examples": eval_hit_patient[:20],
        },
        "lesion_leakage": {
            "shared_lesions": len(shared_lesions),
            "pool_images_involved": sum(len(v) for v in by_lesion.values()),
            "eval_images_involved": len(eval_hit_lesion),
            "eval_images_involved_frac": len(eval_hit_lesion) / len(eval_set),
            "eval_examples": eval_hit_lesion[:20],
        },
        "pool_by_dataset": {
            name: sum(1 for d in pool.values() if d == name) for name in sorted(set(pool.values()))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="cruda/cases.jsonl defining the evaluation split")
    parser.add_argument("--pool-manifest", type=Path, default=DEFAULT_POOL_MANIFEST)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    patient, lesion, diagnostics = load_identity_maps(DEFAULT_SOURCES)
    pool = load_pool(args.pool_manifest)
    eval_ids = load_eval(args.cases)
    result = summarize(pool, eval_ids, patient, lesion)
    result["identifier_diagnostics"] = diagnostics

    # An identifier that does not group cannot refute leakage. Say so before any
    # collision count, so a zero is never read as independence.
    unusable = [d for d in diagnostics if not d["usable_for_grouping"]]
    result["measurability"] = {
        "patient": result["patient_comparison_status"] != "not_measurable",
        "lesion": result["lesion_comparison_status"] != "not_measurable",
    }
    result["leakage_measurable"] = all(result["measurability"].values())
    result["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*DEFAULT_SOURCES.values(), args.pool_manifest, args.cases]
    }
    print("DIAGNOSTICO DE LOS IDENTIFICADORES")
    for diag in diagnostics:
        verdict = "USABLE" if diag["usable_for_grouping"] else "INSERVIBLE"
        print(
            f"  [{verdict}] {diag['name']}: {diag['distinct']} distintos / "
            f"{diag['rows']} filas ({diag['distinct_ratio']:.1%}) - {diag['reason']}"
        )
    if unusable:
        print(
            "\nADVERTENCIA: "
            + ", ".join(d["name"] for d in unusable)
            + " no agrupa imagenes. Cualquier conteo de colisiones apoyado en esa "
            "columna da cero por construccion y NO acredita independencia."
        )
    print()

    cov = result["coverage"]
    print(f"pool: {result['pool_size']}  eval: {result['eval_size']}")
    print(f"solapamiento directo de image_id (control): {result['direct_image_id_overlap']}")
    print("\nCOBERTURA DE IDENTIFICADORES NO PLACEHOLDER (no acredita procedencia clinica)")
    print(
        f"  pool con patient_id: {cov['pool_with_patient_id']} "
        f"({cov['pool_patient_coverage']:.1%})"
    )
    print(
        f"  pool con lesion_id:  {cov['pool_with_lesion_id']} "
        f"({cov['pool_lesion_coverage']:.1%})"
    )
    print(
        f"  eval con patient_id: {cov['eval_with_patient_id']} "
        f"({cov['eval_patient_coverage']:.1%})"
    )
    print(
        f"  eval con lesion_id:  {cov['eval_with_lesion_id']} "
        f"({cov['eval_lesion_coverage']:.1%})"
    )

    for label, key in (("PACIENTE", "patient_leakage"), ("LESION", "lesion_leakage")):
        block = result[key]
        status = result["patient_comparison_status" if key == "patient_leakage" else "lesion_comparison_status"]
        print(f"\nCOINCIDENCIAS POR {label}: {status}")
        if status == "not_measurable":
            print("  Sin identificadores comparables en ambos conjuntos; cero NO interpretable.")
            continue
        print(f"  identidades compartidas: {block['shared_patients' if key == 'patient_leakage' else 'shared_lesions']}")
        print(f"  imagenes del pool involucradas: {block['pool_images_involved']}")
        print(
            f"  imagenes del eval involucradas: {block['eval_images_involved']} "
            f"({block['eval_images_involved_frac']:.2%} del split)"
        )
        if block["eval_examples"]:
            print(f"  ejemplos: {', '.join(block['eval_examples'][:8])}")

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
