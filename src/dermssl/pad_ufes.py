"""PAD-UFES-20 metadata, label mapping and phototype bins, as registered in docs/30.

Everything here is fixed before the data is downloaded: the expected counts
come from Pacheco et al. 2020 (Data in Brief 32:106221, Table 1), the label
mapping and the readings from docs/30 2.2, and the phototype bins from
docs/30 2.5. The parser refuses a metadata file that does not match the
published dataset instead of evaluating on whatever arrived.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: Columns the loader reads. ``fitspatrick`` is spelled as in the dataset.
REQUIRED_COLUMNS = ("patient_id", "lesion_id", "img_id", "diagnostic", "fitspatrick")
#: The article guarantees these are never blank.
ALWAYS_PRESENT = ("patient_id", "lesion_id", "img_id", "diagnostic")

EXPECTED_IMAGES = 2298
EXPECTED_PATIENTS = 1373
EXPECTED_LESIONS = 1641
EXPECTED_DIAGNOSTIC_COUNTS = {"ACK": 730, "BCC": 845, "MEL": 52, "NEV": 244, "SCC": 192, "SEK": 235}

#: docs/30 2.2. SCC includes Bowen's disease; SEK -> bkl is exploratory only.
PAD_TO_HAM = {"BCC": "bcc", "MEL": "mel", "NEV": "nv", "ACK": "akiec", "SCC": "akiec", "SEK": "bkl"}

#: docs/30 2.2: which HAM classes each reading keeps, and whether it carries a verdict.
READINGS = {
    "primary": ("bcc", "mel", "nv"),
    "secondary": ("akiec", "bcc", "mel", "nv"),
    "exploratory": ("akiec", "bcc", "bkl", "mel", "nv"),
}
READINGS_WITH_VERDICT = ("primary", "secondary")

#: docs/30 2.5: phototype bins, fixed before seeing the data.
FST_MISSING = "sin_fst"
FST_BINS = ("I", "II", "III-VI", FST_MISSING)
FST_VALID = range(1, 7)
_FST_BLANKS = frozenset({"", "nan", "none", "unk", "na"})


class PadMetadataError(ValueError):
    """The metadata does not match the published PAD-UFES-20."""


@dataclass(frozen=True)
class PadRecord:
    img_id: str
    patient_id: str
    lesion_id: str
    diagnostic: str
    ham_label: str
    fst: int | None
    fst_status: str  # "ok" | "missing" | "out_of_range"

    @property
    def fst_bin(self) -> str:
        return fst_bin(self.fst)


def parse_fst(raw: str | None) -> tuple[int | None, str]:
    """``(value, status)``. Blanks and UNK are missing; 7 or 2.5 are out of range.

    Out-of-range values are counted apart from blanks so the manifest shows
    how many there were instead of folding them silently into "missing".
    """
    text = "" if raw is None else str(raw).strip()
    if text.lower() in _FST_BLANKS:
        return None, "missing"
    try:
        number = float(text)
    except ValueError:
        return None, "out_of_range"
    if not number.is_integer() or int(number) not in FST_VALID:
        return None, "out_of_range"
    return int(number), "ok"


def fst_bin(value: int | None) -> str:
    if value is None:
        return FST_MISSING
    if value == 1:
        return "I"
    if value == 2:
        return "II"
    return "III-VI"


def image_key(img_id: str) -> str:
    """``PAT_1_2_3.png`` and ``PAT_1_2_3`` name the same image."""
    name = img_id.strip()
    return name[:-4] if name.lower().endswith(".png") else name


def _record(row: dict, line: int) -> PadRecord:
    blanks = [c for c in ALWAYS_PRESENT if not (row.get(c) or "").strip()]
    if blanks:
        raise PadMetadataError(f"fila {line}: {blanks} vacias; el articulo dice que siempre estan")
    diagnostic = row["diagnostic"].strip().upper()
    if diagnostic not in PAD_TO_HAM:
        raise PadMetadataError(f"fila {line}: diagnostico {diagnostic!r} fuera de {sorted(PAD_TO_HAM)}")
    fst, status = parse_fst(row.get("fitspatrick"))
    return PadRecord(
        img_id=image_key(row["img_id"]), patient_id=row["patient_id"].strip(),
        lesion_id=row["lesion_id"].strip(), diagnostic=diagnostic,
        ham_label=PAD_TO_HAM[diagnostic], fst=fst, fst_status=status,
    )


def parse_metadata(path: Path) -> list[PadRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise PadMetadataError(f"{path}: faltan las columnas {missing}")
        records = [_record(row, line) for line, row in enumerate(reader, start=2)]
    duplicated = [k for k, n in Counter(r.img_id for r in records).items() if n > 1]
    if duplicated:
        raise PadMetadataError(f"img_id repetidos: {duplicated[:5]}")
    return records


def count_discrepancies(records: list[PadRecord]) -> list[str]:
    """Every way the metadata departs from the published dataset (empty = matches)."""
    found = []
    counts = {
        "imagenes": (len(records), EXPECTED_IMAGES),
        "pacientes": (len({r.patient_id for r in records}), EXPECTED_PATIENTS),
        "lesiones": (len({r.lesion_id for r in records}), EXPECTED_LESIONS),
    }
    found += [f"{name}: {have}, el articulo dice {want}" for name, (have, want) in counts.items() if have != want]
    diagnostics = Counter(r.diagnostic for r in records)
    found += [f"{dx}: {diagnostics.get(dx, 0)}, el articulo dice {want}"
              for dx, want in EXPECTED_DIAGNOSTIC_COUNTS.items() if diagnostics.get(dx, 0) != want]
    owners: dict[str, set[str]] = {}
    for record in records:
        owners.setdefault(record.lesion_id, set()).add(record.patient_id)
    shared = sorted(lesion for lesion, patients in owners.items() if len(patients) > 1)
    if shared:
        found.append(f"{len(shared)} lesiones con mas de un paciente, p. ej. {shared[:3]}")
    return found


def check_published_counts(records: list[PadRecord]) -> None:
    problems = count_discrepancies(records)
    if problems:
        raise PadMetadataError("el metadata no es el PAD-UFES-20 publicado:\n  " + "\n  ".join(problems))


def fst_summary(records: list[PadRecord]) -> dict:
    return {
        "by_bin": {b: sum(r.fst_bin == b for r in records) for b in FST_BINS},
        "by_status": dict(Counter(r.fst_status for r in records)),
    }
