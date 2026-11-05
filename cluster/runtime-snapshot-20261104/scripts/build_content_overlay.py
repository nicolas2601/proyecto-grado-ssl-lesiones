"""Build a reversible, image-only cross-role content-exclusion overlay."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config/cross-role-content-plan-v1.json"
AUDIT_DIR = ROOT / "results/cross-role-content-audit-v1"
OUTPUT = ROOT / "results/content-exclusion-overlay-v1"
ELIGIBLE_ROLES = ("source_fit", "source_dev", "ham_fit", "ham_dev", "pad_adapt")
PROTECTED_ROLES = ("ham_holdout", "pad_test")
IMAGE_FIELDS = ("image_id", "image_path", "identity_component_id")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def role_pair(left, right):
    return tuple(sorted((left, right)))


def phash_exclusion_pairs(edges, conflicts, exempt_pairs):
    excluded = set()
    for edge in edges:
        if edge["distance"] > 4:
            continue
        for left in edge["left"]:
            for right in edge["right"]:
                pair = tuple(sorted((left["image_path"], right["image_path"])))
                if role_pair(left["role"], right["role"]) in conflicts and pair not in exempt_pairs:
                    excluded.add(pair)
    return excluded


def exact_rgb_endpoints(groups, conflicts):
    excluded = set()
    for group in groups:
        members = group["members"]
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if role_pair(left["role"], right["role"]) in conflicts:
                    excluded.update((left["image_path"], right["image_path"]))
    return excluded


def exclusion_keys(phash_edges, rgb_groups, conflicts, exempt_pairs):
    near_pairs = phash_exclusion_pairs(phash_edges, conflicts, exempt_pairs)
    raw_ids = set()
    paths = exact_rgb_endpoints(rgb_groups, conflicts)
    reasons = {}
    for edge in phash_edges:
        if edge["distance"] > 4:
            continue
        for left in edge["left"]:
            for right in edge["right"]:
                pair = tuple(sorted((left["image_path"], right["image_path"])))
                if role_pair(left["role"], right["role"]) in conflicts and pair not in exempt_pairs:
                    paths.update((left["image_path"], right["image_path"]))
                    for row in (left, right):
                        reasons.setdefault((row["image_id"], row["image_path"]), set()).add("phash_le4_disjoint_edge")
    for group in rgb_groups:
        members = group["members"]
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if role_pair(left["role"], right["role"]) in conflicts:
                    for row in (left, right):
                        reasons.setdefault((row["image_id"], row["image_path"]), set()).add("rgb_exact_disjoint")
    # ID and path propagation is global across role aliases, not component closure.
    raw_ids.update(image_id for image_id, _ in reasons)
    return {"image_ids": raw_ids, "image_paths": paths}


def filter_rows(rows, excluded):
    ids, paths = excluded["image_ids"], excluded["image_paths"]
    components = excluded.get("identity_component_ids", set())
    return [row for row in rows if row["image_id"] not in ids and row["image_path"] not in paths
            and row.get("identity_component_id") not in components]


def verify_plan(plan, summary, plan_bytes):
    if summary.get("status") != "COMPLETE" or summary.get("plan_sha256") != hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError("Audit summary is not COMPLETE for this exact plan")
    if summary.get("near_content_pairs") != summary.get("near_content_pairs_exported"):
        raise ValueError("Audit summary has incomplete pHash edge export")
    for source in plan["manifests"]:
        if summary.get("role_images", {}).get(source["role"]) != sum(row["role"] == source["role"] for row in plan["records"]):
            raise ValueError(f"Audit count does not match plan for {source['role']}")


def run(image_fields_path, output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite overlay output: {output}")
    plan_bytes = PLAN_PATH.read_bytes()
    plan = json.loads(plan_bytes)
    summary_path = AUDIT_DIR / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    verify_plan(plan, summary, plan_bytes)
    input_paths = [PLAN_PATH, summary_path, AUDIT_DIR / "phash-candidates.jsonl",
                   AUDIT_DIR / "rgb-groups.jsonl", AUDIT_DIR / "visual-sample.json",
                   AUDIT_DIR / "adjudication.json", Path(image_fields_path)]
    input_hashes = {str(path): sha256(path) for path in input_paths}
    visual = json.loads((AUDIT_DIR / "visual-sample.json").read_text())
    adjudication = json.loads((AUDIT_DIR / "adjudication.json").read_text())
    if adjudication.get("status") != "sample_reviewed_training_blocked":
        raise ValueError("Visual adjudication is not in the reviewed state")
    exempt_pairs = {tuple(sorted((visual["pairs"][index][side]["image_path"]
                                  for side in ("left", "right")))) for index in (10, 14)}
    conflicts = {role_pair(*pair) for pair in plan["conflict_role_pairs"]}
    edges = load_jsonl(AUDIT_DIR / "phash-candidates.jsonl")
    rgb_groups = load_jsonl(AUDIT_DIR / "rgb-groups.jsonl")
    excluded = exclusion_keys(edges, rgb_groups, conflicts, exempt_pairs)
    fields = load_jsonl(image_fields_path)
    expected_counts = {role: summary["role_images"][role] for role in ELIGIBLE_ROLES}
    by_role = {role: [] for role in ELIGIBLE_ROLES}
    seen = {role: set() for role in ELIGIBLE_ROLES}
    for row in fields:
        if set(row) != set(IMAGE_FIELDS) | {"role"}:
            raise ValueError("Input must contain only image_id, image_path, identity_component_id")
        for value in row.values():
            if not isinstance(value, str) or not value:
                raise ValueError("Canonical image identity fields must be nonblank strings")
        if row["role"] not in by_role or row["image_id"] in seen[row["role"]]:
            raise ValueError("Expected unique IDs and only the five eligible roles")
        seen[row["role"]].add(row["image_id"])
        by_role[row["role"]].append(row)
    if not fields or any(len(by_role[r]) != expected_counts[r] for r in ELIGIBLE_ROLES):
        raise ValueError("Image-only canonical field rows do not match audit role counts")
    approved = {(row["role"], row["image_id"]): (row["image_path"], row["identity_component_id"])
                for row in plan["records"]}
    for role in ELIGIBLE_ROLES:
        if any(approved.get((role, row["image_id"])) != (row["image_path"], row["identity_component_id"])
               for row in by_role[role]):
            raise ValueError(f"Image fields do not exactly match approved plan rows for {role}")
    protected = {row["image_id"] for row in plan["records"] if row["role"] in PROTECTED_ROLES}
    protected_paths = {row["image_path"] for row in plan["records"] if row["role"] in PROTECTED_ROLES}
    protected_components = {row["identity_component_id"] for row in plan["records"]
                            if row["role"] in PROTECTED_ROLES}
    safe_excluded = {"image_ids": excluded["image_ids"] | protected,
                     "image_paths": excluded["image_paths"] | protected_paths,
                     "identity_component_ids": protected_components}
    manifests = {}
    role_counts = {}
    for role in ELIGIBLE_ROLES:
        original = by_role[role]
        clean = filter_rows(original, safe_excluded)
        if any(row["identity_component_id"] in protected_components for row in clean):
            raise AssertionError(f"Protected identity component survived in {role}")
        if any(row["image_id"] in protected or row["image_path"] in protected_paths for row in clean):
            raise AssertionError(f"Protected image survived in {role}")
        role_counts[role] = {"original": len(original), "excluded": len(original) - len(clean),
                             "remaining": len(clean)}
        manifests[role] = clean
    for path in input_paths:
        if sha256(path) != input_hashes[str(path)]:
            raise ValueError(f"Input changed during overlay generation: {path}")
    output.mkdir(parents=False, exist_ok=False)
    output_hashes = {}
    for role, rows in manifests.items():
        manifest_path = output / f"{role}.jsonl"
        with manifest_path.open("x") as stream:
            for row in rows:
                stream.write(json.dumps({field: row[field] for field in IMAGE_FIELDS}, sort_keys=True) + "\n")
            if not rows:
                stream.write("")
        output_hashes[manifest_path.name] = sha256(manifest_path)
    excluded_path = output / "excluded.jsonl"
    with excluded_path.open("x") as stream:
        unique = {}
        for row in plan["records"]:
            if row["image_id"] in safe_excluded["image_ids"] or row["image_path"] in safe_excluded["image_paths"]:
                unique[(row["role"], row["image_id"], row["image_path"])] = row
        for key in sorted(unique):
            row = unique[key]
            stream.write(json.dumps({**{field: row[field] for field in ("role", "image_id", "image_path", "identity_component_id")},
                                     "reason": "protected_role_identity_or_path" if row["role"] in PROTECTED_ROLES else "cross_role_content_or_protected_identity"}, sort_keys=True) + "\n")
    output_hashes[excluded_path.name] = sha256(excluded_path)
    receipt = {"status": "COMPLETE", "scope": "Reversible image-only derived overlay; canonical manifests untouched",
               "input_sha256": input_hashes, "output_sha256": output_hashes,
               "per_role_counts": role_counts,
               "excluded_candidate_phash_pairs": len(phash_exclusion_pairs(edges, conflicts, exempt_pairs)),
               "exempt_physical_path_pairs": [list(pair) for pair in sorted(exempt_pairs)],
               "protected_roles": list(PROTECTED_ROLES), "protected_component_policy": "Exclude any candidate row whose identity_component_id occurs in ham_holdout or pad_test; do not remove further component members unless this protected-identity filter requires it.",
               "protected_components_removed_from_eligible": sum(1 for role in ELIGIBLE_ROLES for row in by_role[role]
                    if row["identity_component_id"] in protected_components and row["image_id"] not in excluded["image_ids"] and row["image_path"] not in excluded["image_paths"]),
               "limits": ["No embedding full-pool audit was conducted.", "Fixed pHash distance <=4 has limited coverage and may miss transformed duplicates.", "Zero exact-RGB disjoint conflicts does not exonerate transformed content leakage.", "This overlay cannot guarantee all protected image content is absent.", "No training, external upload, canonical edit, or patient-component closure was performed."]}
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_fields", type=Path, help="Image-only JSONL from the five eligible canonical roles")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.image_fields, args.output), indent=2))
