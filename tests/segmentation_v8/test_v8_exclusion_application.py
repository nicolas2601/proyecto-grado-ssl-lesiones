"""CLI contract fixtures for exclusion review; execute only in UNABIA."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "apply_v8_exclusion_review.py"


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def source_reference(path):
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


class ExclusionApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pool = self.root / "original_records.jsonl"
        self.register = self.root / "register.json"
        self.queue = self.root / "nearest.json"
        self.report = self.root / "phash_report.json"
        self.evaluation_zip = self.root / "evaluation.zip"
        self.readiness = self.root / "readiness.review.json"
        self.output = self.root / "application"

        excluded = []
        proposals = []
        evaluation_images = []
        for number in range(1, 1918):
            image_id = f"ISIC_{number:07d}"
            member = f"evaluation/{image_id}.jpg"
            derivative = number > 583
            record = self.pool_record(image_id + ("_downsampled" if derivative else ""))
            excluded.append(record)
            proposals.append({
                "eval_id": image_id, "eval_member": member,
                "match_kind": "terminal_downsampled_derivative" if derivative else "exact_stem",
                "pool_record": record, "status": "proposed_required_not_applied",
                "exclusion_applied": False,
            })
            evaluation_images.append({"eval_id": image_id, "eval_member": member})

        # A shared numeric suffix alone must never remove a different full ISIC ID.
        self.retained = [self.pool_record("ISIC_10000001"),
                         self.pool_record("ISIC_9000001_downsampled"),
                         self.pool_record("ISIC_9000002")]
        records = excluded[:700] + self.retained[:1] + excluded[700:] + self.retained[1:]
        self.pool.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        self.pairs = [
            self.pair("ISIC_0000001", self.retained[0], 0),
            self.pair("ISIC_0000001", self.retained[1], 2),
            self.pair("ISIC_0000002", self.retained[2], 4),
        ]
        write_json(self.report, {"screen": {"candidate_pairs": self.pairs}})
        # These are provenance-only bytes: application must not reopen images or recompute pHash.
        self.evaluation_zip.write_bytes(b"fixture: saved evaluation provenance only\n")
        nearest_pairs = [{key: value for key, value in self.pairs[index].items()
                          if key not in {"status", "exclusion_applied"}}
                         for index in (0, 2)]
        self.queue_payload = {"closest_pair_records": nearest_pairs,
                              "nearest_distance_counts": {"0": 1, "4": 1},
                              "source_report_sha256": source_reference(self.report)["sha256"]}
        write_json(self.queue, self.queue_payload)
        write_json(self.register, {
            "schema": "v8_exclusion_review_register.v1",
            "artifact_role": "audit_only_exclusion_proposals_and_unadjudicated_candidates",
            "executable": False, "training_consumable": False, "manifest_frozen": False,
            "execution_enabled": False, "launch_ready": False, "training_started": False,
            "patient_lesion_independence": "unresolved",
            "patient_lesion_groups_preserved": False,
            "base_id_exclusion_proposals": proposals,
            "phash_candidate_pairs": self.pairs,
            "phash_pool_records": self.retained,
            "evaluation_images": evaluation_images,
            "consistency": {"all_checks_passed": True, "observed": {
                "pool_records": 1920, "evaluation_images": 1917,
                "base_id_overlaps": 1917, "exact_stem_overlaps": 583,
                "downsampled_overlaps": 1334, "phash_candidate_pairs": 3,
                "phash_unique_evaluation_ids": 2, "phash_unique_pool_paths": 3}},
            "sources": {"pool_records": source_reference(self.pool),
                        "evaluation_zip": source_reference(self.evaluation_zip),
                        "phash_report": source_reference(self.report)},
        })
        write_json(self.readiness, {"data": {
            "step_1_freeze_gate_review": {"exclusion_review_register": {
                "artifact_path": str(self.register),
                "artifact_bytes": self.register.stat().st_size,
                "artifact_sha256": source_reference(self.register)["sha256"],
                "consistency_all_checks_passed": True,
                "observed_counts": {"pool_records": 1920, "base_id_overlaps": 1917,
                                    "exact_stem_overlaps": 583, "downsampled_overlaps": 1334,
                                    "phash_candidate_pairs": 3},
                "exclusions_applied": False}},
            "isic2018_pool_phash_screen": {"nearest_review_queue": {
                **source_reference(self.queue), "rows": 2,
                "status": "review_queue_not_adjudicated", "decisions_applied": False}},
        }})

    @staticmethod
    def pool_record(image_id):
        return {"image_id": image_id, "dataset": "isic2019",
                "image_path": f"/unavailable/fixture-images/{image_id}.jpg",
                "image_sha256": "a" * 64, "bytes": 42, "width": 224, "height": 224}

    @staticmethod
    def pair(eval_id, record, distance):
        return {"eval_id": eval_id, "eval_member": f"evaluation/{eval_id}.jpg",
                "train_id": record["image_id"].removesuffix("_downsampled"),
                "train_path": record["image_path"], "hamming_distance_63bit": distance,
                "exact_file_sha256_match": False, "status": "review_pending",
                "exclusion_applied": False}

    def apply(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--readiness", str(self.readiness),
             "--output-dir", str(self.output)],
            capture_output=True, text=True, check=False, timeout=30,
        )

    def test_filters_full_base_ids_and_preserves_every_pending_similarity_pair(self):
        # Catches suffix-only filtering, nearest-only ledgers, and inferred pHash decisions.
        originals = {path: path.read_bytes() for path in
                     (self.pool, self.register, self.queue, self.report,
                      self.evaluation_zip, self.readiness)}

        result = self.apply()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        filtered = self.output / "training_pool_base_id_filtered.jsonl"
        self.assertTrue(filtered.is_file(), "Application must write a provisional filtered pool")
        rows = [json.loads(line) for line in filtered.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, self.retained)
        with (self.output / "phash_adjudication_ledger.csv").open(newline="", encoding="utf-8") as handle:
            ledger = list(csv.DictReader(handle))
        self.assertEqual(len(ledger), 3)
        by_key = {(row["eval_id"], row["train_path"]): row for row in ledger}
        self.assertEqual(set(by_key), {(pair["eval_id"], pair["train_path"]) for pair in self.pairs})
        for index, pair in enumerate(self.pairs):
            row = by_key[(pair["eval_id"], pair["train_path"])]
            self.assertEqual(row["train_id"], pair["train_id"])
            self.assertEqual(int(row["hamming_distance_63bit"]), pair["hamming_distance_63bit"])
            self.assertEqual(row["status"], "review_pending")
            self.assertEqual(row["in_nearest_queue"].lower(), "true" if index in (0, 2) else "false")
        receipt = json.loads((self.output / "receipt.json").read_text(encoding="utf-8"))
        self.assertIs(receipt["manifest_frozen"], False)
        self.assertIs(receipt["training_consumable"], False)
        self.assertIs(receipt["patient_independence_accredited"], False)
        self.assertEqual(receipt["counts"]["pool_records_before"], 1920)
        self.assertEqual(receipt["counts"]["pool_records_after"], 3)
        self.assertEqual(receipt["counts"]["base_id_exclusions_applied"], 1917)
        self.assertEqual(receipt["counts"]["phash_candidate_pairs"], 3)
        self.assertEqual(receipt["counts"]["phash_pending_pairs"], 3)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original, f"Input modified: {path.name}")

    def test_rejects_changed_registered_inputs_without_producing_outputs(self):
        # Catches ignored source hashes, including stale readiness-to-register pins.
        for path in (self.register, self.pool, self.queue, self.report, self.evaluation_zip):
            with self.subTest(source=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tampered\n")
                try:
                    result = self.apply()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("sha256", result.stderr.lower(), result.stdout + result.stderr)
                    self.assertFalse(self.output.exists(), "Hash rejection must not publish partial outputs")
                finally:
                    path.write_bytes(original)

    def test_refuses_existing_output_directory_without_modifying_contents(self):
        # Catches overwriting or mixing a new application with an earlier review.
        self.output.mkdir()
        sentinel = self.output / "receipt.json"
        sentinel.write_bytes(b"existing audit receipt\n")

        result = self.apply()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exist", result.stderr.lower(), result.stdout + result.stderr)
        self.assertEqual(sentinel.read_bytes(), b"existing audit receipt\n")
        self.assertEqual([path.name for path in self.output.iterdir()], ["receipt.json"])


if __name__ == "__main__":
    unittest.main()
