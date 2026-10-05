"""The official PAD-UFES-20 files: download, integrity and image paths (docs/30 2.1).

Sizes and sha256 were read from the public Mendeley API on 2026-09-28
(``data.mendeley.com/public-api/datasets/zr7vgbcyr2``, version 1). A file
whose size or digest differs is deleted and the run stops: evaluating on a
truncated archive gives a number that looks like a result.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

DATASET_DOI = "10.17632/zr7vgbcyr2.1"
LICENSE = "CC BY 4.0 (http://creativecommons.org/licenses/by/4.0)"
DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/{file_id}/file_downloaded"
USER_AGENT = "dermssl-pad-ufes-download/1.0 (research; CC BY 4.0 dataset)"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class OfficialFile:
    name: str
    file_id: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return DOWNLOAD_URL.format(file_id=self.file_id)


OFFICIAL_FILES = (
    OfficialFile("imgs_part_1.zip", "1cc2f71f-20a2-412d-b746-a9b9bc20c966", 1245184680,
                 "0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca"),
    OfficialFile("imgs_part_2.zip", "559a60ed-5504-475d-996c-6a8bc253b5e7", 1126646990,
                 "e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b"),
    OfficialFile("imgs_part_3.zip", "34dcdf8e-e5f1-4b35-aa0b-5135051ff852", 1220565093,
                 "ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c"),
    OfficialFile("metadata.csv", "fa850265-57da-48f0-ba3e-998b3e44b1f6", 316209,
                 "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"),
)
METADATA_NAME = "metadata.csv"


class IntegrityError(RuntimeError):
    """A file is not the one the dataset publishes."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, official: OfficialFile) -> None:
    """Size first (cheap, catches truncation), then the digest."""
    if not path.is_file():
        raise IntegrityError(f"{path} no existe")
    size = path.stat().st_size
    if size != official.size:
        raise IntegrityError(f"{path.name}: {size} bytes, Mendeley publica {official.size}")
    digest = sha256_of(path)
    if digest != official.sha256:
        raise IntegrityError(f"{path.name}: sha256 {digest[:16]}..., Mendeley publica {official.sha256[:16]}...")


Opener = Callable[[str], BinaryIO]


def _default_opener(url: str) -> BinaryIO:
    # Mendeley answers 403 to urllib's default "Python-urllib/x.y" agent.
    from urllib.request import Request, urlopen  # noqa: PLC0415

    request = Request(url, headers={"User-Agent": USER_AGENT})
    return urlopen(request, timeout=60)  # noqa: S310 - fixed https URL above


def fetch(official: OfficialFile, dest_dir: Path, opener: Opener = _default_opener) -> Path:
    """Download one file to ``dest_dir`` unless a verified copy is already there.

    Written to ``<name>.part`` and renamed only after it verifies, so an
    interrupted download never leaves a file with the final name.
    """
    target = Path(dest_dir) / official.name
    if target.is_file():
        verify(target, official)
        return target
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    with opener(official.url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, CHUNK_BYTES)
    try:
        verify(partial, official)
    except IntegrityError:
        partial.unlink()
        raise
    partial.rename(target)
    return target


def safe_extract(archive: Path, dest_dir: Path) -> int:
    """Extract ``archive`` refusing any member that would land outside ``dest_dir``."""
    root = Path(dest_dir).resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise IntegrityError(f"{archive.name}: {member!r} sale del directorio destino")
        zf.extractall(root)
        return len(zf.namelist())


def index_images(root: Path) -> dict[str, Path]:
    """``image key -> path`` for every PNG under ``root``; a name found twice is refused."""
    from dermssl.pad_ufes import image_key  # noqa: PLC0415

    index: dict[str, Path] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() != ".png" or not path.is_file():
            continue
        key = image_key(path.name)
        if key in index:
            raise IntegrityError(f"{key} aparece dos veces: {index[key]} y {path}")
        index[key] = path
    return index


def resolve_images(img_ids: list[str], index: dict[str, Path]) -> dict[str, Path]:
    missing = [i for i in img_ids if i not in index]
    if missing:
        raise IntegrityError(f"{len(missing)} imagenes del metadata sin archivo, p. ej. {missing[:3]}")
    return {i: index[i] for i in img_ids}
