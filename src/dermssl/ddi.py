"""DDI (Diverse Dermatology Images) metadata, label mapping and the malignant/benign
primary reading, as planned for the zero-shot evaluation of v7 against a real,
expert-assigned phototype.

Source: Daneshjou et al., "Disparities in dermatology AI performance on a diverse,
clinically characterized image set", Science Advances 8, eabq6147 (2022). Distributed
by Stanford AIMI under the Stanford University Dataset Research Use Agreement
(personal, non-commercial, no redistribution) - images never enter this repository,
only hashes and derived metrics do, same pattern as PAD-UFES-20 (``pad_ufes.py``).

``skin_tone`` in ``ddi_metadata.csv`` is Fitzpatrick skin type GROUPED into three
bins by the dataset's own publishers, assigned by consensus of two board-certified
dermatologists via chart review - not self-estimated by any ITA instrument. ``12``
= FST I-II, ``34`` = FST III-IV, ``56`` = FST V-VI.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ZIP_SHA256 = "f9099a51fd451932c5923d2feaf025f2a3e28fae628fc5ae477eb418fe837e9f"
ZIP_NAME = "ddidiversedermatologyimages.zip"
LICENSE = "Stanford University Dataset Research Use Agreement (personal, non-commercial, no redistribution)"
SOURCE_URL = "https://ddi-dataset.github.io"

REQUIRED_COLUMNS = ("DDI_ID", "DDI_file", "skin_tone", "malignant", "disease")
EXPECTED_IMAGES = 656

#: skin_tone bin -> (benign, malignant), published on the dataset's download page,
#: verified against ddi_metadata.csv on 2026-11-04.
EXPECTED_SKIN_TONE_COUNTS = {"12": (159, 49), "34": (167, 74), "56": (159, 48)}
SKIN_TONE_LABEL = {"12": "FST I-II", "34": "FST III-IV", "56": "FST V-VI"}
SKIN_TONE_BINS = ("12", "34", "56")

#: DDI's 78 free-text ``disease`` labels that map unambiguously onto a HAM10000
#: class, per the taxonomy each HAM10000 class is defined by (Tschandl et al. 2018).
#: Deliberately NOT exhaustive: nevus variants whose equivalence to HAM's ``nv``
#: definition is not confidently verified here (blue-nevus, nevus-lipomatosus-
#: superficialis, pigmented/atypical-spindle-cell-nevus-of-reed) are left OUT
#: rather than guessed, same convention as SEK in PAD-UFES (docs/30 2.2): a
#: disease absent from this table is unmapped, not silently forced into a class.
DISEASE_TO_HAM: dict[str, str] = {
    # nv - melanocytic nevi, dysplastic nevi included per Tschandl et al. 2018.
    "melanocytic-nevi": "nv",
    "dysplastic-nevus": "nv",
    # bkl - benign keratosis-like lesions (seborrheic keratosis, solar lentigo,
    # lichen-planus-like keratosis).
    "seborrheic-keratosis": "bkl",
    "seborrheic-keratosis-irritated": "bkl",
    "solar-lentigo": "bkl",
    "lichenoid-keratosis": "bkl",
    "benign-keratosis": "bkl",
    # bcc - basal cell carcinoma, all variants.
    "basal-cell-carcinoma": "bcc",
    "basal-cell-carcinoma-nodular": "bcc",
    "basal-cell-carcinoma-superficial": "bcc",
    # akiec - actinic keratoses and intraepithelial carcinoma (Bowen's disease).
    # Invasive SCC has NO analogue in HAM's 7 classes (same partial-mapping
    # declaration as PAD's SCC -> akiec, docs/30 2.2) and is excluded here.
    "actinic-keratosis": "akiec",
    "squamous-cell-carcinoma-in-situ": "akiec",
    # df - dermatofibroma.
    "dermatofibroma": "df",
    # vasc - vascular lesions.
    "pyogenic-granuloma": "vasc",
    "angioma": "vasc",
    "arteriovenous-hemangioma": "vasc",
    "glomangioma": "vasc",
    # mel - melanoma, all subtypes/stages, same convention HAM10000 itself uses.
    "melanoma": "mel",
    "melanoma-in-situ": "mel",
    "melanoma-acral-lentiginous": "mel",
    "nodular-melanoma-(nm)": "mel",
}

#: The primary reading needs no mapping at all: ``malignant`` is native to DDI.
#: This table says which HAM10000 classes count as malignant/premalignant when
#: the probe's 7-class probability is collapsed to the binary DDI asks for.
#: akiec is premalignant/in-situ by definition, not unambiguously malignant -
#: declared as a modeling choice with the no-akiec reading as a sensitivity check.
HAM_MALIGNANT = frozenset({"mel", "bcc", "akiec"})
HAM_MALIGNANT_STRICT = frozenset({"mel", "bcc"})  # sensitivity reading, excludes akiec
HAM_BENIGN = frozenset({"nv", "bkl", "df", "vasc"})
HAM_CLASSES = HAM_MALIGNANT | HAM_BENIGN


class DdiMetadataError(ValueError):
    """The metadata does not match the published DDI dataset."""


@dataclass(frozen=True)
class DdiRecord:
    ddi_id: str
    file_name: str
    skin_tone: str  # "12" | "34" | "56"
    malignant: bool
    disease: str

    @property
    def ham_label(self) -> str | None:
        """``None`` when ``disease`` has no verified HAM10000 analogue (see DISEASE_TO_HAM)."""
        return DISEASE_TO_HAM.get(self.disease)

    @property
    def skin_tone_label(self) -> str:
        return SKIN_TONE_LABEL[self.skin_tone]


def _parse_malignant(raw: str) -> bool:
    text = raw.strip().lower()
    if text not in ("true", "false"):
        raise DdiMetadataError(f"malignant {raw!r} no es True/False")
    return text == "true"


def _record(row: dict, line: int) -> DdiRecord:
    blanks = [c for c in REQUIRED_COLUMNS if not (row.get(c) or "").strip()]
    if blanks:
        raise DdiMetadataError(f"fila {line}: {blanks} vacias")
    skin_tone = row["skin_tone"].strip()
    if skin_tone not in SKIN_TONE_BINS:
        raise DdiMetadataError(f"fila {line}: skin_tone {skin_tone!r} fuera de {SKIN_TONE_BINS}")
    return DdiRecord(
        ddi_id=row["DDI_ID"].strip(), file_name=row["DDI_file"].strip(),
        skin_tone=skin_tone, malignant=_parse_malignant(row["malignant"]),
        disease=row["disease"].strip(),
    )


def parse_metadata(path: Path) -> list[DdiRecord]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DdiMetadataError(f"{path}: faltan las columnas {missing}")
        records = [_record(row, line) for line, row in enumerate(reader, start=2)]
    duplicated = [k for k, n in Counter(r.file_name for r in records).items() if n > 1]
    if duplicated:
        raise DdiMetadataError(f"DDI_file repetidos: {duplicated[:5]}")
    return records


def count_discrepancies(records: list[DdiRecord]) -> list[str]:
    """Every way the metadata departs from the published DDI dataset (empty = matches)."""
    found = []
    if len(records) != EXPECTED_IMAGES:
        found.append(f"imagenes: {len(records)}, la pagina publica {EXPECTED_IMAGES}")
    for bin_name, (ben, mal) in EXPECTED_SKIN_TONE_COUNTS.items():
        sub = [r for r in records if r.skin_tone == bin_name]
        have_ben = sum(1 for r in sub if not r.malignant)
        have_mal = sum(1 for r in sub if r.malignant)
        if (have_ben, have_mal) != (ben, mal):
            found.append(
                f"skin_tone {bin_name} ({SKIN_TONE_LABEL[bin_name]}): "
                f"benigno={have_ben} maligno={have_mal}, la pagina publica benigno={ben} maligno={mal}"
            )
    return found


def check_published_counts(records: list[DdiRecord]) -> None:
    problems = count_discrepancies(records)
    if problems:
        raise DdiMetadataError("el metadata no es el DDI publicado:\n  " + "\n  ".join(problems))


def skin_tone_summary(records: list[DdiRecord]) -> dict:
    return {
        bin_name: {
            "label": SKIN_TONE_LABEL[bin_name],
            "n": sum(1 for r in records if r.skin_tone == bin_name),
            "benign": sum(1 for r in records if r.skin_tone == bin_name and not r.malignant),
            "malignant": sum(1 for r in records if r.skin_tone == bin_name and r.malignant),
        }
        for bin_name in SKIN_TONE_BINS
    }


def ham_mapping_coverage(records: list[DdiRecord]) -> dict:
    """How many records map to a HAM10000 class, and the unmapped diseases by count."""
    mapped = [r for r in records if r.ham_label is not None]
    unmapped_diseases = Counter(r.disease for r in records if r.ham_label is None)
    return {
        "n_total": len(records),
        "n_mapped": len(mapped),
        "n_unmapped": len(records) - len(mapped),
        "by_ham_class": dict(Counter(r.ham_label for r in mapped)),
        "unmapped_diseases": dict(unmapped_diseases),
    }
