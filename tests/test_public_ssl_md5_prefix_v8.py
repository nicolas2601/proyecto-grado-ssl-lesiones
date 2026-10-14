"""Only new publisher-checksum paths; no previous suites or real weight files."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_public_ssl_staging_v8 import staging, Response, ROOT, URL


class PublisherPrefix(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.plan = json.loads((ROOT / "plan-segmentacion/metodos-v8/public-weights.plan.json").read_text())
        self.item = copy.deepcopy(self.plan["artifacts"][1])
        self.item.update(url=URL, expected_bytes=8, max_bytes=100)
        self.md5 = hashlib.md5(b"abcdefgh", usedforsecurity=False).hexdigest()

    def transfer(self):
        with patch.object(staging.urllib.request, "build_opener") as make:
            make.return_value.open.return_value = Response()
            return staging.download(self.item, self.folder, 1, 1, 60)

    def test_matching_published_prefix_keeps_full_local_hashes(self):
        self.item["published_md5_prefix"] = self.md5[:8]
        target, meta = self.transfer()
        self.assertEqual(meta["md5"], self.md5)
        self.assertEqual(meta["sha256"], hashlib.sha256(b"abcdefgh").hexdigest())
        self.assertEqual(meta["integrity"], "matches_published_md5_prefix")
        self.assertEqual(target.read_bytes(), b"abcdefgh")

    def test_wrong_prefix_preserves_partial_without_promoting(self):
        self.item["published_md5_prefix"] = "0" * 8
        with self.assertRaisesRegex(ValueError, "MD5 prefix"):
            self.transfer()
        self.assertFalse((self.folder / "source.pth.tar").exists())
        self.assertEqual((self.folder / "source.pth.tar.part").read_bytes(), b"abcdefgh")

    def test_prefix_is_not_treated_as_a_complete_md5(self):
        self.plan["artifacts"][1]["published_md5_prefix"] = self.md5
        path = self.folder / "invalid-plan.json"
        path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            staging.read_plan(path)


if __name__ == "__main__":
    unittest.main()
