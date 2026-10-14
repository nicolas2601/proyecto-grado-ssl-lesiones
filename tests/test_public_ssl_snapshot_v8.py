"""Only the new exported-source identity path; no models, network or old suites."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("staging_snapshot", ROOT / "scripts/stage_public_ssl_weights_v8.py")
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


class ExportedSourceIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / "scripts/prepare_public_ssl_weights_v8.py"
        self.adapter.parent.mkdir()
        self.adapter.write_text("# mock adapter\n")
        self.source = self.root / "src/mock_source.py"
        self.source.parent.mkdir()
        self.source.write_text("# mock imported source\n")
        self.manifest = self.root / "adapter-source-manifest.json"
        revision = "ab" * 20
        self.manifest.write_text(json.dumps({"schema_version": 1, "producer_revision": revision,
            "files": {str(p.relative_to(self.root)): staging.digest(p) for p in (self.adapter, self.source)}}))
        self.plan = {"adapter_revision": revision, "adapter_sha256": staging.digest(self.adapter),
                     "adapter_source_manifest_sha256": staging.digest(self.manifest)}

    def identity(self):
        with patch.object(staging, "ROOT", self.root), patch.object(staging, "ADAPTER", self.adapter):
            return staging.adapter_identity(self.plan)

    def test_complete_pinned_export_records_real_source_revision(self):
        result = self.identity()
        self.assertEqual(result["producer_revision"], self.plan["adapter_revision"])
        self.assertEqual(result["source_manifest_sha256"], staging.digest(self.manifest))

    def test_changed_imported_source_is_rejected(self):
        self.source.write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "file changed"):
            self.identity()

    def test_unrecorded_source_is_rejected(self):
        (self.source.parent / "unrecorded.py").write_text("# unrecorded\n")
        with self.assertRaisesRegex(ValueError, "unrecorded files"):
            self.identity()

    def test_replaced_manifest_is_rejected(self):
        self.manifest.write_text("{}")
        with self.assertRaisesRegex(ValueError, "pinned manifest hash"):
            self.identity()


if __name__ == "__main__":
    unittest.main()
