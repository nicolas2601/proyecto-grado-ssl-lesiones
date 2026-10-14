"""New staging I/O regressions only: mocked HTTP, temporary bytes, no ML jobs."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("staging", ROOT / "scripts/stage_public_ssl_weights_v8.py")
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)
URL = "https://dl.fbaipublicfiles.com/example.pth"


class Response:
    def __init__(self, body=b"abcdefgh", status=200, headers=None, fail_after=None):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body)), "ETag": '"v1"', **(headers or {})}
        self.fail_after = fail_after

    def geturl(self):
        return URL

    def read(self, count):
        if self.fail_after is not None and self.body.tell() >= self.fail_after:
            raise OSError("simulated interrupted connection")
        return self.body.read(count)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class StagingIO(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.item = copy.deepcopy(json.loads((ROOT / "plan-segmentacion/metodos-v8/public-weights.plan.json").read_text())["artifacts"][1])
        self.item.update(url=URL, expected_bytes=8, max_bytes=100)

    def transfer(self, response):
        with patch.object(staging.urllib.request, "build_opener") as make:
            make.return_value.open.return_value = response
            result = staging.download(self.item, self.folder, 1, 1, 60)
            return result, make.return_value.open.call_args

    def interrupt(self):
        with patch.object(staging, "CHUNK", 4):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.transfer(Response(fail_after=4))
        self.assertEqual((self.folder / "source.pth.tar.part").read_bytes(), b"abcd")

    def test_interrupted_download_resumes_exact_range(self):
        self.interrupt()
        (target, meta), call = self.transfer(Response(b"efgh", 206, {"Content-Range": "bytes 4-7/8"}))
        request = call.args[0]
        self.assertEqual(request.get_header("Range"), "bytes=4-")
        self.assertEqual(request.get_header("If-range"), '"v1"')
        self.assertEqual(target.read_bytes(), b"abcdefgh")
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["integrity"], "local_sha256_only")

    def test_ignored_range_and_changed_object_preserve_partial(self):
        for response in (Response(), Response(b"efgh", 206, {"Content-Range": "bytes 4-7/8", "ETag": '"v2"'})):
            with self.subTest(headers=response.headers):
                if not (self.folder / "source.pth.tar.part").exists():
                    self.interrupt()
                with self.assertRaises(ValueError):
                    self.transfer(response)
                self.assertEqual((self.folder / "source.pth.tar.part").read_bytes(), b"abcd")

    def test_bad_checksum_is_never_promoted(self):
        self.item["expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.transfer(Response())
        self.assertFalse((self.folder / "source.pth.tar").exists())
        self.assertTrue((self.folder / "source.pth.tar.part").exists())

    def test_completed_source_is_not_redownloaded_and_tampering_fails(self):
        (target, _), _ = self.transfer(Response())
        with patch.object(staging.urllib.request, "build_opener") as make:
            staging.download(self.item, self.folder, 1, 1, 60)
            make.assert_not_called()
        target.write_bytes(b"changed!")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.transfer(Response())

    def test_partial_tampering_fails_before_network(self):
        self.interrupt()
        (self.folder / "source.pth.tar.part").write_bytes(b"evil")
        with patch.object(staging.urllib.request, "build_opener") as make:
            with self.assertRaisesRegex(ValueError, "durable progress"):
                staging.download(self.item, self.folder, 1, 1, 60)
            make.assert_not_called()

    def test_crash_after_complete_transfer_recovers_without_network(self):
        target = self.folder / "source.pth.tar"
        part = self.folder / "source.pth.tar.part"
        part.write_bytes(b"abcdefgh")
        sha = hashlib.sha256(part.read_bytes()).hexdigest()
        staging.atomic_json(self.folder / "transfer.json", {
            "artifact_identity": staging.identity(self.item), "committed_bytes": 8,
            "total_bytes": 8, "partial_sha256": sha, "complete": False})
        with patch.object(staging.urllib.request, "build_opener") as make:
            staging.download(self.item, self.folder, 1, 1, 60)
            make.assert_not_called()
        self.assertEqual(target.read_bytes(), b"abcdefgh")

    def test_html_and_login_redirect_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "document"):
            self.transfer(Response(headers={"Content-Type": "text/html"}))
        self.assertFalse((self.folder / "source.pth.tar.part").exists())
        with self.assertRaises(ValueError):
            staging.PublicRedirect().redirect_request(None, None, 302, "", {}, "https://accounts.google.com/login")

    def test_default_plan_has_no_side_effects(self):
        workspace = self.folder / "not-created"
        args = ["stage", "--plan", str(ROOT / "plan-segmentacion/metodos-v8/public-weights.plan.json"),
                "--workspace", str(workspace)]
        with patch("sys.argv", args), patch("sys.stdout", io.StringIO()), \
                patch.object(staging.urllib.request, "build_opener") as make:
            self.assertEqual(staging.main(), 0)
            make.assert_not_called()
        self.assertFalse(workspace.exists())

    def test_workspace_inside_any_checkout_is_rejected(self):
        checkout = self.folder / "protected-checkout"
        (checkout / ".git").mkdir(parents=True)
        workspace = checkout / "staging"
        args = ["stage", "--plan", str(ROOT / "plan-segmentacion/metodos-v8/public-weights.plan.json"),
                "--workspace", str(workspace), "--download"]
        with patch("sys.argv", args), patch.object(staging.urllib.request, "build_opener") as make:
            with self.assertRaisesRegex(ValueError, "outside the checkout"):
                staging.main()
            make.assert_not_called()
        self.assertFalse(workspace.exists())

    def test_review_block_does_not_prevent_another_artifact(self):
        first = copy.deepcopy(self.item)
        second = copy.deepcopy(self.item)
        second["id"] = "second"
        second["review"] = dict.fromkeys(("reviewed_by", "reviewed_at", "license_id", "license_evidence_url",
                                          "artifact_license_scope", "source_selection_evidence",
                                          "normalization_evidence_url"), "mock-review")
        second["review"].update(approved=True, input_normalization="imagenet")
        plan = json.loads((ROOT / "plan-segmentacion/metodos-v8/public-weights.plan.json").read_text())
        plan["artifacts"] = [first, second]
        plan_file = self.folder / "plan.json"
        plan_file.write_text(json.dumps(plan))
        workspace = self.folder / "staging"
        args = ["stage", "--plan", str(plan_file), "--workspace", str(workspace), "--download", "--reserve-bytes", "1"]
        with patch("sys.argv", args), patch("sys.stdout", io.StringIO()), \
                patch.object(staging.urllib.request, "build_opener") as make:
            make.return_value.open.return_value = Response()
            self.assertEqual(staging.main(), 2)
        ledger = json.loads((workspace / "weights-ledger.json").read_text())["artifacts"]
        self.assertEqual(ledger[first["id"]]["status"], "blocked")
        self.assertEqual(ledger["second"]["status"], "downloaded")

    def test_tf_extracts_only_explicit_complete_shards(self):
        source = self.folder / "source.zip"
        item = {"checkpoint_prefix": "ResNet50_1x/model.ckpt-225206", "max_unpacked_bytes": 100}
        prefix = item["checkpoint_prefix"]
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr(prefix + ".index", b"index")
            archive.writestr(prefix + ".data-00000-of-00001", b"data")
            archive.writestr("../escape", b"not extracted")
        out = self.folder / "extracted"
        out.mkdir()
        result = staging.extract_checkpoint(item, source, out, 1)
        self.assertEqual(result.name, "model.ckpt-225206")
        self.assertEqual(len(list(out.iterdir())), 2)
        self.assertFalse((self.folder / "escape").exists())

    def test_http_403_preserves_source_and_does_not_retry(self):
        with patch.object(staging.urllib.request, "build_opener") as make:
            make.return_value.open.side_effect = urllib.error.HTTPError(URL, 403, "Forbidden", {}, None)
            with self.assertRaises(urllib.error.HTTPError):
                staging.download(self.item, self.folder, 1, 1, 60)
            self.assertEqual(make.return_value.open.call_count, 1)
        self.assertFalse((self.folder / "source.pth.tar").exists())

    def fake_adapter(self, args, **kwargs):
        target = Path(args[args.index("--output") + 1])
        target.write_bytes(b"mock converted tensor file")
        provenance = {"protocol": "public_ssl_initialization_v8", "method": "ssl_mocov1",
                      "checkpoint_sha256": staging.digest(target), "input_normalization": "imagenet",
                      "source_license": "mock-license", "source_sha256": self.transfer_meta["sha256"]}
        target.with_suffix(".pt.json").write_text(json.dumps(provenance))

    def test_conversion_commits_pair_and_skips_completed_pair(self):
        self.item["review"].update(license_id="mock-license", input_normalization="imagenet")
        (source, transfer), _ = self.transfer(Response())
        self.transfer_meta = transfer
        code = {"producer_revision": "mock", "adapter_sha256": "mock"}
        with patch.object(staging.subprocess, "run", side_effect=self.fake_adapter) as run:
            output = staging.convert({}, self.item, source, transfer, self.folder, code, 1)
            self.assertTrue((output / "encoder.pt").exists())
            self.assertTrue((output / "encoder.pt.json").exists())
            self.assertEqual(staging.convert({}, self.item, source, transfer, self.folder, code, 1), output)
            self.assertEqual(run.call_count, 1)
        (output / "encoder.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed on disk"):
            staging.convert({}, self.item, source, transfer, self.folder, code, 1)

    def test_failed_conversion_never_publishes_pair(self):
        import subprocess
        self.item["review"].update(license_id="mock-license", input_normalization="imagenet")
        (source, transfer), _ = self.transfer(Response())
        with patch.object(staging.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "mock")):
            with self.assertRaises(subprocess.CalledProcessError):
                staging.convert({}, self.item, source, transfer, self.folder, {"producer_revision": "mock"}, 1)
        self.assertFalse((self.folder / "converted").exists())
        self.assertTrue(list(self.folder.glob("conversion-*")))


if __name__ == "__main__":
    unittest.main()
