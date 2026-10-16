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


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


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
        self.phash_receipt = self.root / "phash_adjudication" / "receipt.json"
        self.phash_pair_decisions = self.root / "phash_adjudication" / "pair_decisions.jsonl"
        self.phash_path_decisions = self.root / "phash_adjudication" / "training_path_decisions.jsonl"
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

    def apply(self, output=None, include_phash=False, receipt_sha256=None):
        output = output or self.output
        command = [sys.executable, str(SCRIPT), "--readiness", str(self.readiness),
                   "--output-dir", str(output)]
        if include_phash:
            command.extend(["--phash-adjudication-receipt", str(self.phash_receipt),
                            "--phash-adjudication-receipt-sha256",
                            receipt_sha256 or source_reference(self.phash_receipt)["sha256"]])
        return subprocess.run(
            command,
            capture_output=True, text=True, check=False, timeout=30,
        )

    def make_phash_receipt(self):
        self.phash_receipt.parent.mkdir()
        pair_decisions = [{**pair,
                           "disposition": "exclude_training_path_conservatively",
                           "duplicate_identity_status": "unverified",
                           "decision_basis": "conservative_eligibility_from_saved_phash_screen"}
                          for pair in self.pairs]
        path_decisions = []
        records_by_path = {row["image_path"]: row for row in self.retained}
        for path in sorted({pair["train_path"] for pair in self.pairs}):
            record = records_by_path[path]
            path_decisions.append({
                "train_path": path,
                "train_id": record["image_id"].removesuffix("_downsampled"),
                "dataset": record["dataset"],
                "disposition": "exclude_training_path_conservatively",
                "duplicate_identity_status": "unverified",
                "exclusion_applied": False,
            })
        write_jsonl(self.phash_pair_decisions, pair_decisions)
        write_jsonl(self.phash_path_decisions, path_decisions)
        base_paths = {proposal["pool_record"]["image_path"] for proposal in
                      json.loads(self.register.read_text(encoding="utf-8"))["base_id_exclusion_proposals"]}
        phash_paths = {row["train_path"] for row in path_decisions}
        pool_records = [json.loads(line) for line in self.pool.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        pool_count = len(pool_records)
        receipt = {
            "schema": "v8_phash_conservative_adjudication.v1",
            "executable": False, "training_consumable": False, "manifest_frozen": False,
            "execution_enabled": False, "launch_ready": False, "training_started": False,
            "patient_independence_accredited": False,
            "patient_lesion_independence": "unresolved",
            "artifact_role": "audit_only_conservative_path_eligibility_decisions",
            "policy": "exclude_training_path_conservatively",
            "decision_scope": "path_eligibility_only_no_visual_or_clinical_identity_adjudication",
            "duplicate_identity_status": "unverified",
            "exclusions_applied_to_inventory": False,
            "sources": {"register": source_reference(self.register),
                        "phash_report": source_reference(self.report),
                        "pool_records": source_reference(self.pool)},
            "counts": {
                "pool_records": pool_count,
                "candidate_pair_decisions": len(pair_decisions),
                "candidate_evaluation_ids": len({row["eval_id"] for row in pair_decisions}),
                "phash_excluded_training_paths": len(phash_paths),
                "base_id_exclusion_proposals": len(base_paths),
                "union_excluded_training_paths": len(base_paths | phash_paths),
                "remaining_candidate_count": pool_count - len(base_paths | phash_paths),
            },
            "per_source": {"isic2019": {
                "before": pool_count,
                "excluded_by_union": len(base_paths | phash_paths),
                "remaining_candidate_count": pool_count - len(base_paths | phash_paths),
            }},
            "outputs": {"pairs": {**source_reference(self.phash_pair_decisions),
                                   "rows": len(pair_decisions)},
                        "paths": {**source_reference(self.phash_path_decisions),
                                  "rows": len(path_decisions)}},
        }
        write_json(self.phash_receipt, receipt)
        return receipt

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

    def test_applies_receipt_pinned_phash_paths_without_accrediting_identity(self):
        # Catches using pHash similarity as duplicate/patient identity evidence or freezing a manifest.
        self.unmatched_retained = [self.pool_record("ISIC_9000003"),
                                   self.pool_record("ISIC_9000004")]
        pool_rows = [json.loads(line) for line in self.pool.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        pool_rows.extend(self.unmatched_retained)
        write_jsonl(self.pool, pool_rows)
        register = json.loads(self.register.read_text(encoding="utf-8"))
        register["sources"]["pool_records"] = source_reference(self.pool)
        register["consistency"]["observed"]["pool_records"] = len(pool_rows)
        write_json(self.register, register)
        readiness = json.loads(self.readiness.read_text(encoding="utf-8"))
        register_pin = readiness["data"]["step_1_freeze_gate_review"]["exclusion_review_register"]
        register_pin["artifact_bytes"] = self.register.stat().st_size
        register_pin["artifact_sha256"] = source_reference(self.register)["sha256"]
        register_pin["observed_counts"]["pool_records"] = len(pool_rows)
        write_json(self.readiness, readiness)
        self.make_phash_receipt()
        originals = {path: path.read_bytes() for path in
                     (self.pool, self.register, self.queue, self.report,
                      self.evaluation_zip, self.readiness, self.phash_receipt,
                      self.phash_pair_decisions, self.phash_path_decisions)}

        result = self.apply(include_phash=True)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        filtered = self.output / "training_pool_base_id_phash_eligibility_filtered.jsonl"
        self.assertTrue(filtered.is_file())
        retained_rows = [json.loads(line) for line in filtered.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(retained_rows, self.unmatched_retained)
        self.assertEqual([row["image_id"] for row in retained_rows],
                         ["ISIC_9000003", "ISIC_9000004"])
        with (self.output / "phash_adjudication_ledger.csv").open(newline="", encoding="utf-8") as handle:
            ledger = list(csv.DictReader(handle))
        self.assertEqual(len(ledger), len(self.pairs))
        self.assertTrue(all(row["status"] == "review_pending" for row in ledger))
        self.assertTrue(all(row["decision"] == "exclude_training_path_conservatively" for row in ledger))
        self.assertTrue(all(row["exclusion_applied"] == "true" for row in ledger))
        receipt = json.loads((self.output / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["schema"], "v8_provisional_exclusion_application.v2")
        self.assertEqual(receipt["application_mode"], "base_id_plus_conservative_phash_path_eligibility")
        self.assertIs(receipt["manifest_frozen"], False)
        self.assertIs(receipt["training_consumable"], False)
        self.assertIs(receipt["execution_enabled"], False)
        self.assertIs(receipt["execution_ready"], False)
        self.assertIs(receipt["launch_ready"], False)
        self.assertIs(receipt["training_started"], False)
        self.assertIs(receipt["patient_independence_accredited"], False)
        self.assertIs(receipt["clinical_independence_accredited"], False)
        self.assertEqual(receipt["patient_lesion_independence"], "unresolved")
        self.assertEqual(receipt["counts"]["pool_records_after"], 2)
        self.assertEqual(receipt["counts"]["phash_exclusions_applied"], 3)
        self.assertEqual(receipt["counts"]["union_exclusions_applied"], 1920)
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original, f"Input modified: {path.name}")

    def test_receipt_mode_records_but_does_not_consume_stale_readiness_pins(self):
        self.make_phash_receipt()
        readiness = json.loads(self.readiness.read_text(encoding="utf-8"))
        register_pin = readiness["data"]["step_1_freeze_gate_review"]["exclusion_review_register"]
        register_pin["artifact_sha256"] = "0" * 64
        register_pin["artifact_bytes"] += 1
        write_json(self.readiness, readiness)

        result = self.apply(include_phash=True)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads((self.output / "receipt.json").read_text(encoding="utf-8"))
        reconciliation = receipt["readiness_pin_reconciliation"]
        self.assertIs(reconciliation["readiness_register_pin_matches_current"], False)
        self.assertEqual(reconciliation["historical_readiness_pins"][
            "exclusion_review_register"]["sha256"], "0" * 64)
        self.assertTrue(any(row["source"] == "exclusion_review_register"
                            for row in reconciliation["discrepancies"]))
        self.assertEqual(reconciliation["mode"], "receipt_pinned_current_sources")

    def test_rejects_receipt_with_wrong_external_hash_before_output(self):
        self.make_phash_receipt()

        result = self.apply(include_phash=True, receipt_sha256="0" * 64)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sha256", result.stderr.lower(), result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_pair_decision_that_disagrees_with_pinned_report(self):
        receipt = self.make_phash_receipt()
        decisions = [json.loads(line) for line in self.phash_pair_decisions.read_text(
            encoding="utf-8").splitlines()]
        decisions[0]["train_path"] = "/different/image.jpg"
        write_jsonl(self.phash_pair_decisions, decisions)
        receipt["outputs"]["pairs"] = {
            **source_reference(self.phash_pair_decisions), "rows": len(decisions)}
        write_json(self.phash_receipt, receipt)

        result = self.apply(include_phash=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pair", result.stderr.lower(), result.stdout + result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_path_decision_that_disagrees_with_original_pool(self):
        receipt = self.make_phash_receipt()
        decisions = [json.loads(line) for line in self.phash_path_decisions.read_text(
            encoding="utf-8").splitlines()]
        decisions[0]["train_id"] = "ISIC_9999999"
        write_jsonl(self.phash_path_decisions, decisions)
        receipt["outputs"]["paths"] = {
            **source_reference(self.phash_path_decisions), "rows": len(decisions)}
        write_json(self.phash_receipt, receipt)

        result = self.apply(include_phash=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("path decisions", result.stderr.lower(), result.stdout + result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
