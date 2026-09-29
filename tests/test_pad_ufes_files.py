"""docs/30 2.1: the PAD-UFES-20 download is verified against Mendeley, offline.

No test opens the network: the opener is injected and serves bytes from memory.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from dermssl import pad_ufes_files as pf


def _official(payload: bytes, name: str = "metadata.csv") -> pf.OfficialFile:
    return pf.OfficialFile(name, "file-id", len(payload), hashlib.sha256(payload).hexdigest())


def _opener(payload: bytes, calls: list | None = None):
    def opener(url):
        if calls is not None:
            calls.append(url)
        return io.BytesIO(payload)
    return opener


class TestTheOfficialList:
    def test_the_four_published_files_with_their_size(self):
        names = [f.name for f in pf.OFFICIAL_FILES]
        assert names == ["imgs_part_1.zip", "imgs_part_2.zip", "imgs_part_3.zip", pf.METADATA_NAME]
        assert sum(f.size for f in pf.OFFICIAL_FILES) == 3_592_712_972
        assert all(len(f.sha256) == 64 for f in pf.OFFICIAL_FILES)

    def test_urls_point_at_the_dataset(self):
        assert all("zr7vgbcyr2" in f.url and f.file_id in f.url for f in pf.OFFICIAL_FILES)
        assert pf.DATASET_DOI.endswith("zr7vgbcyr2.1")


class TestFetch:
    def test_a_good_download_lands_with_its_name(self, tmp_path):
        payload = b"patient_id,img_id\n"
        path = pf.fetch(_official(payload), tmp_path, opener=_opener(payload))
        assert path.read_bytes() == payload
        assert not list(tmp_path.glob("*.part"))

    def test_a_truncated_download_is_deleted_and_refused(self, tmp_path):
        official = _official(b"0123456789")
        with pytest.raises(pf.IntegrityError, match="bytes"):
            pf.fetch(official, tmp_path, opener=_opener(b"01234"))
        assert list(tmp_path.iterdir()) == []

    def test_same_size_other_bytes_is_refused_by_the_digest(self, tmp_path):
        official = _official(b"aaaa")
        with pytest.raises(pf.IntegrityError, match="sha256"):
            pf.fetch(official, tmp_path, opener=_opener(b"bbbb"))
        assert list(tmp_path.iterdir()) == []

    def test_a_verified_copy_is_not_downloaded_again(self, tmp_path):
        payload = b"x" * 10
        (tmp_path / "metadata.csv").write_bytes(payload)
        calls: list = []
        pf.fetch(_official(payload), tmp_path, opener=_opener(payload, calls))
        assert calls == []

    def test_a_corrupt_existing_copy_is_refused_not_trusted(self, tmp_path):
        (tmp_path / "metadata.csv").write_bytes(b"y" * 10)
        with pytest.raises(pf.IntegrityError):
            pf.fetch(_official(b"x" * 10), tmp_path, opener=_opener(b"x" * 10))


class TestExtractAndIndex:
    def _zip(self, path: Path, members: dict[str, bytes]) -> Path:
        with zipfile.ZipFile(path, "w") as zf:
            for name, data in members.items():
                zf.writestr(name, data)
        return path

    def test_extract_and_resolve(self, tmp_path):
        archive = self._zip(tmp_path / "a.zip", {"images/PAT_1_2_3.png": b"p", "images/PAT_4_5_6.png": b"q"})
        assert pf.safe_extract(archive, tmp_path / "out") == 2
        index = pf.index_images(tmp_path / "out")
        assert set(index) == {"PAT_1_2_3", "PAT_4_5_6"}
        assert pf.resolve_images(["PAT_1_2_3"], index)["PAT_1_2_3"].read_bytes() == b"p"

    def test_a_member_escaping_the_directory_is_refused(self, tmp_path):
        archive = self._zip(tmp_path / "a.zip", {"../evil.png": b"x"})
        with pytest.raises(pf.IntegrityError, match="sale del directorio"):
            pf.safe_extract(archive, tmp_path / "out")
        assert not (tmp_path / "evil.png").exists()

    def test_an_image_present_twice_is_refused(self, tmp_path):
        for part in ("p1", "p2"):
            (tmp_path / part).mkdir()
            (tmp_path / part / "PAT_1_2_3.png").write_bytes(b"x")
        with pytest.raises(pf.IntegrityError, match="dos veces"):
            pf.index_images(tmp_path)

    def test_an_image_without_file_is_refused(self, tmp_path):
        with pytest.raises(pf.IntegrityError, match="sin archivo"):
            pf.resolve_images(["PAT_9_9_9"], {})


class TestTheDefaultOpener:
    def test_it_does_not_send_the_urllib_user_agent_mendeley_refuses(self, monkeypatch):
        seen = {}

        def fake_urlopen(request, timeout):
            seen["ua"] = request.get_header("User-agent")
            seen["url"] = request.full_url
            return io.BytesIO(b"")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        pf._default_opener("https://data.mendeley.com/x")
        assert seen["url"] == "https://data.mendeley.com/x"
        assert seen["ua"] and "urllib" not in seen["ua"].lower()
