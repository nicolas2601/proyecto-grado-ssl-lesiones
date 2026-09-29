"""scripts/pad_ufes_download.py end to end, offline, on tiny synthetic files."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from dermssl import pad_ufes_files as pf
from scripts import pad_ufes_download as dl


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture()
def tiny_dataset(monkeypatch):
    payloads = {
        "imgs_part_1.zip": _zip_bytes({"imgs_part_1/PAT_1_1_1.png": b"a"}),
        "metadata.csv": b"patient_id,img_id\nPAT_1,PAT_1_1_1.png\n",
    }
    files = tuple(pf.OfficialFile(name, f"id-{name}", len(data), hashlib.sha256(data).hexdigest())
                  for name, data in payloads.items())
    monkeypatch.setattr(pf, "OFFICIAL_FILES", files)
    by_url = {f.url: payloads[f.name] for f in files}
    return lambda url: io.BytesIO(by_url[url])


def test_downloads_verifies_and_extracts(tmp_path, tiny_dataset):
    assert dl.main(["--dest", str(tmp_path), "--extract"], opener=tiny_dataset) == 0
    assert (tmp_path / dl.IMAGES_DIR / "imgs_part_1" / "PAT_1_1_1.png").read_bytes() == b"a"
    assert (tmp_path / "metadata.csv").is_file()


def test_a_corrupt_file_exits_non_zero_and_leaves_nothing(tmp_path, tiny_dataset):
    corrupt = lambda url: io.BytesIO(b"\0" * len(tiny_dataset(url).read()))  # noqa: E731
    assert dl.main(["--dest", str(tmp_path)], opener=corrupt) == 1
    assert not (tmp_path / "imgs_part_1.zip").exists()
