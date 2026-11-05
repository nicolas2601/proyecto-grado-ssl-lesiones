"""Read-only cross-role image-content audit; never consumes diagnosis metadata."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import struct
import time

from PIL import Image, ImageOps
from audit_near_duplicates import phash

RADIUS = 4
MAX_EXPORTED_NEAR_PAIRS = 250000


def fingerprint(path):
    try:
        before = Path(path).stat()
        raw = Path(path).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            pixels = struct.pack(">II", *rgb.size) + rgb.tobytes()
            bits = phash(rgb)
        after = Path(path).stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Image changed during fingerprinting")
        return {"path": path, "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                "phash": sum(int(bit) << index for index, bit in enumerate(bits)),
                "size": after.st_size, "mtime_ns": after.st_mtime_ns}
    except Exception as error:
        return {"path": path, "error": f"{type(error).__name__}: {error}"}


class HashIndex:
    def __init__(self):
        self.root = None

    def add(self, value):
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def query(self, value, radius):
        pending = [self.root] if self.root else []
        matches = []
        while pending:
            current, children = pending.pop()
            distance = (current ^ value).bit_count()
            if distance <= radius:
                matches.append(current)
            pending.extend(child for delta, child in children.items()
                           if distance - radius <= delta <= distance + radius)
        return matches


def pair_key(left, right):
    return "|".join(sorted((left, right)))


def matrix(roles):
    return {pair_key(a, b): {"groups": 0, "pair_observations": 0, "left_ids": set(), "right_ids": set()}
            for a, b in combinations(sorted(roles), 2)}


def register(group, counts):
    by_role = defaultdict(set)
    for row in group:
        by_role[row["role"]].add(row["image_id"])
    for a, b in combinations(sorted(by_role), 2):
        cell = counts[pair_key(a, b)]
        cell["groups"] += 1
        cell["pair_observations"] += len(by_role[a]) * len(by_role[b])
        cell["left_ids"].update(by_role[a])
        cell["right_ids"].update(by_role[b])


def conflict(group, conflicts):
    roles = {row["role"] for row in group}
    return any(a in roles and b in roles for a, b in conflicts)


def affected(group, removed):
    for row in group:
        removed[row["role"]].add(row["image_id"])


def sample(group, reason, conflicts, samples):
    for left, right in combinations(group, 2):
        if tuple(sorted((left["role"], right["role"]))) not in conflicts:
            continue
        priority = 0 if {left["role"], right["role"]} & {"ham_holdout", "pad_test"} else 1
        candidate = {"priority": priority, "reason": reason, "left": left, "right": right}
        signature = {left["image_path"], right["image_path"]}
        if not any({item["left"]["image_path"], item["right"]["image_path"]} == signature for item in samples):
            samples.append(candidate)
            samples.sort(key=lambda item: item["priority"])
            del samples[12:]
        return


def exact_groups(groups, output, kind, roles, conflicts, removed, samples):
    counts = matrix(roles)
    with (output / f"{kind}-groups.jsonl").open("x") as stream:
        for digest, group in groups.items():
            if len({row["role"] for row in group}) < 2:
                continue
            register(group, counts)
            is_conflict = conflict(group, conflicts)
            stream.write(json.dumps({"hash": digest, "conflict": is_conflict, "members": group}) + "\n")
            if is_conflict:
                affected(group, removed)
                sample(group, kind, conflicts, samples)
    return counts


def near_groups(rgb_groups, fingerprints, output, roles, conflicts, removed, samples):
    counts, tree, buckets = matrix(roles), HashIndex(), defaultdict(list)
    total, written = 0, 0
    with (output / "phash-candidates.jsonl").open("x") as stream:
        for digest, group in sorted(rgb_groups.items()):
            value = fingerprints[group[0]["image_path"]]["phash"]
            for matching in tree.query(value, RADIUS):
                for previous_digest in buckets[matching]:
                    previous = rgb_groups[previous_digest]
                    together = previous + group
                    if len({row["role"] for row in together}) < 2:
                        continue
                    total += 1
                    register(together, counts)
                    is_conflict = conflict(together, conflicts)
                    distance = (value ^ matching).bit_count()
                    if written < MAX_EXPORTED_NEAR_PAIRS:
                        stream.write(json.dumps({"left_rgb": previous_digest, "right_rgb": digest,
                                                 "distance": distance, "conflict": is_conflict,
                                                 "left": previous, "right": group}) + "\n")
                        written += 1
                    if is_conflict:
                        affected(together, removed)
                        sample(together, f"phash_distance_{distance}", conflicts, samples)
            tree.add(value)
            buckets[value].append(digest)
    return counts, total, written


def serial_matrix(counts, expected, conflicts):
    result = {}
    for key, cell in counts.items():
        pair = tuple(key.split("|"))
        relationship = "compatible_roles_or_parent" if pair in expected else (
            "declared_disjoint" if pair in conflicts else "unclassified")
        result[key] = {"groups": cell["groups"], "pair_observations": cell["pair_observations"],
                       "left_unique_ids": len(cell["left_ids"]), "right_unique_ids": len(cell["right_ids"]),
                       "relationship": relationship}
    return result


def impact(roles, originals, removed):
    return {role: {"original_images": len(originals[role]), "candidate_removed_images": len(removed[role]),
                   "remaining_images": len(originals[role] - removed[role])} for role in sorted(roles)}


def validate_sources(plan):
    for source in plan.get("manifests", []):
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Canonical source manifest changed")


def run(plan_path, output):
    started = time.monotonic()
    raw = Path(plan_path).read_bytes()
    plan = json.loads(raw)
    validate_sources(plan)
    observations, originals = defaultdict(list), defaultdict(set)
    for original in plan["records"]:
        row = {key: str(original[key]) for key in ("role", "image_id", "image_path")}
        if not all(value.strip() for value in row.values()) or row["image_id"] in originals[row["role"]]:
            raise ValueError("Expected unique nonblank IDs within each role")
        row["image_path"] = str(Path(row["image_path"]).resolve())
        if "identity_component_id" in original:
            row["identity_component_id"] = str(original["identity_component_id"])
        originals[row["role"]].add(row["image_id"])
        observations[row["image_path"]].append(row)
    roles = sorted(originals)
    expected = {tuple(sorted(pair)) for pair in plan["expected_role_pairs"]}
    conflicts = {tuple(sorted(pair)) for pair in plan["conflict_role_pairs"]}
    if expected & conflicts:
        raise ValueError("A role pair cannot be both containment and disjoint")
    output.mkdir(parents=False, exist_ok=False)
    fingerprints, errors = {}, []
    with ThreadPoolExecutor(max_workers=2) as executor, (output / "fingerprints.jsonl").open("x") as stream:
        for number, item in enumerate(executor.map(fingerprint, sorted(observations)), start=1):
            stream.write(json.dumps(item) + "\n")
            if "error" in item:
                errors.append(item)
            else:
                fingerprints[item["path"]] = item
            if number % 2000 == 0:
                stream.flush()
                print(f"fingerprinted {number}/{len(observations)}, seconds={time.monotonic() - started:.1f}", flush=True)
    grouped = {"byte": defaultdict(list), "rgb": defaultdict(list)}
    for path, item in fingerprints.items():
        for kind in grouped:
            grouped[kind][item[f"{kind}_sha256"]].extend(observations[path])
    removed_exact, samples = defaultdict(set), []
    byte = exact_groups(grouped["byte"], output, "byte", roles, conflicts, defaultdict(set), samples)
    rgb = exact_groups(grouped["rgb"], output, "rgb", roles, conflicts, removed_exact, samples)
    removed_all = defaultdict(set, {role: set(values) for role, values in removed_exact.items()})
    near, total, written = near_groups(grouped["rgb"], fingerprints, output, roles, conflicts, removed_all, samples)
    validate_sources(plan)
    if Path(plan_path).read_bytes() != raw:
        raise ValueError("Audit plan changed during execution")
    report = {"status": "COMPLETE" if not errors and total == written else "INCOMPLETE",
              "scope": "Image-content-only audit across canonical classifier roles; no diagnosis metadata or scoring",
              "plan_sha256": hashlib.sha256(raw).hexdigest(), "manifests": plan.get("manifests", []),
              "role_images": {role: len(originals[role]) for role in roles},
              "distinct_physical_paths": len(observations), "fingerprints_completed": len(fingerprints),
              "distinct_rgb_content": len(grouped["rgb"]), "errors": errors,
              "RGB_definition": "EXIF orientation normalized, RGB uint8; SHA256 of big-endian width,height plus pixel bytes",
              "pHash_definition": "63-bit DCT pHash from normalized RGB; DC excluded; Hamming<=4 fixed before audit",
              "byte": serial_matrix(byte, expected, conflicts), "rgb": serial_matrix(rgb, expected, conflicts),
              "phash": serial_matrix(near, expected, conflicts),
              "near_content_pairs": total, "near_content_pairs_exported": written,
              "proposed_exclusion": {"rgb_exact_conflicts": impact(roles, originals, removed_exact),
                                     "conservative_rgb_and_phash_conflicts": impact(roles, originals, removed_all)},
              "proposal_only": "No roles/IDs/files changed. Remove all observations in declared-disjoint cross-role content groups from both sides, including parent-role aliases. RGB-exact and conservative pHash scenarios are separate; visually adjudicate near-candidates before deciding any exclusion.",
              "count_definition": "Exact groups are content hashes spanning roles. pHash groups are edges joining two different RGB content hashes; unique-ID and pair-observation counts concern their member unions, not independent patients or all pairwise distances.",
              "visual_sample": samples, "duration_seconds": time.monotonic() - started,
              "limits": "No embedding similarity; fixed pHash audit can miss transformed duplicates. No global patient linkage, automatic exclusions, diagnosis access, clinical evaluation or change to Pau's pipelines."}
    with (output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "role_images": report["role_images"], "near_content_pairs": total,
                      "errors": len(errors), "duration_seconds": report["duration_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    raise SystemExit(0 if result["status"] == "COMPLETE" else 2)
