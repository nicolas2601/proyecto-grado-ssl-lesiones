"""Clinical independence gate: accredit only on a complete, bridged identity graph.

Implements the rule recorded for v8: independence is accredited only when the
combined patient-and-lesion identity graph is evaluated over *every* exposure of
the model, groups carry comparable links, no group straddles training and
evaluation, and every exposure manifest is covered.

The gate is BLOCKED by default. Any one of these blocks the claim:

  * an image with no verifiable identity in any exposure
  * an identifier namespace with no documented bridge to another namespace
  * an exposure manifest whose image list is not pinned by a hash
  * an exposure manifest that is declared but absent

Rationale for the default, from CLAIM 2024 item 20: partitions "generally should
be disjoint at the patient level or higher". An absent identifier or a string
that fails to match is UNKNOWN, never independent -- ISIC ids can change between
cohorts, so a non-match proves nothing. Item 12 covers how the missingness
itself must be reported.

Exposures are every set of images the weights ever saw: the unlabeled selection
and refit pools, the PH2 labelled subset, the IMA++ cohort, and the DermLIP
pretraining corpus. Omitting one makes the graph incomplete and the verdict
meaningless, so an uncovered exposure blocks rather than being skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class IdentityGraph:
    """Union-find over images, linked by identifiers qualified with their namespace."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def _find(self, node: str) -> str:
        self._parent.setdefault(node, node)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def link(self, left: str, right: str) -> None:
        left_root, right_root = self._find(left), self._find(right)
        if left_root != right_root:
            self._parent[left_root] = right_root

    def component(self, node: str) -> str:
        return self._find(node)

    def components(self, nodes: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for node in nodes:
            groups.setdefault(self._find(node), []).append(node)
        return groups


def qualified(namespace: str, value: str) -> str:
    """Namespace-qualified identifier. Never compare raw values across sources."""
    return f"{namespace}::{value}"


def build_graph(exposures: dict[str, dict]) -> tuple[IdentityGraph, dict]:
    graph = IdentityGraph()
    identified: set[str] = set()
    namespaces: set[str] = set()

    for name, exposure in exposures.items():
        for image_id, identity in exposure.get("identities", {}).items():
            node = qualified("image", image_id)
            for field in ("patient_id", "lesion_id"):
                value = identity.get(field)
                if not value:
                    continue
                namespace = identity.get(f"{field}_namespace") or name
                namespaces.add(namespace)
                graph.link(node, qualified(f"{namespace}.{field}", str(value)))
                identified.add(image_id)

    return graph, {"identified_images": identified, "namespaces": sorted(namespaces)}


def evaluate(
    exposures: dict[str, dict],
    eval_images: list[str],
    declared_exposures: list[str],
    bridges: list[list[str]],
) -> dict:
    blocks: list[dict] = []

    # 1. Every declared exposure must be present and hash-pinned.
    for name in declared_exposures:
        exposure = exposures.get(name)
        if exposure is None:
            blocks.append({"kind": "exposure_missing", "exposure": name})
            continue
        if not exposure.get("manifest_sha256"):
            blocks.append({"kind": "image_list_not_hash_pinned", "exposure": name})

    graph, info = build_graph(exposures)

    # 2. Every image in every exposure, plus the evaluation split, needs an identity.
    all_images: set[str] = set(eval_images)
    for exposure in exposures.values():
        all_images.update(exposure.get("images", []))
    unidentified = sorted(all_images - info["identified_images"])
    if unidentified:
        blocks.append(
            {
                "kind": "images_without_verifiable_identity",
                "count": len(unidentified),
                "total": len(all_images),
                "examples": unidentified[:10],
            }
        )

    # 3. Every namespace must be bridged to at least one other, or it cannot be
    #    compared across exposures and a non-collision is not evidence.
    bridged = {ns for bridge in bridges for ns in bridge}
    unbridged = [ns for ns in info["namespaces"] if ns not in bridged]
    if len(info["namespaces"]) > 1 and unbridged:
        blocks.append({"kind": "namespace_without_bridge", "namespaces": unbridged})

    # 4. No identity group may straddle training exposures and the evaluation split.
    train_images: set[str] = set()
    for name, exposure in exposures.items():
        if name == "evaluation":
            continue
        train_images.update(exposure.get("images", []))
    train_components = {graph.component(qualified("image", i)) for i in train_images}
    straddling = sorted(
        i for i in eval_images if graph.component(qualified("image", i)) in train_components
    )
    if straddling:
        blocks.append(
            {
                "kind": "identity_group_straddles_train_and_eval",
                "eval_images_affected": len(straddling),
                "examples": straddling[:10],
            }
        )

    return {
        "verdict": "ACCREDITED" if not blocks else "BLOCKED",
        "patient_independence_accredited": not blocks,
        "blocks": blocks,
        "namespaces": info["namespaces"],
        "identified_images": len(info["identified_images"]),
        "images_considered": len(all_images),
        "eval_images": len(eval_images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "spec",
        type=Path,
        help="JSON describing exposures, the evaluation split, declared exposures and bridges",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = evaluate(
        spec["exposures"],
        spec["evaluation_images"],
        spec.get(
            "declared_exposures",
            ["selection_pool", "refit_pool", "ph2", "ima_plusplus", "dermlip_pretraining"],
        ),
        spec.get("namespace_bridges", []),
    )

    print(f"VEREDICTO: {result['verdict']}")
    print(
        f"  imagenes con identidad verificable: {result['identified_images']} "
        f"de {result['images_considered']}"
    )
    print(f"  namespaces: {', '.join(result['namespaces']) or '(ninguno)'}")
    if result["blocks"]:
        print("\nBLOQUEOS:")
        for block in result["blocks"]:
            detail = ", ".join(f"{k}={v}" for k, v in block.items() if k != "kind")
            print(f"  - {block['kind']}: {detail}")

    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    raise SystemExit(0 if result["verdict"] == "ACCREDITED" else 1)


if __name__ == "__main__":
    main()
