"""docs/30 2.1, 2.2 and 2.5: PAD-UFES-20 metadata parsing, fixed before the download.

Synthetic CSVs only: no test touches the real dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from dermssl import pad_ufes as pu

HEADER = ("patient_id", "lesion_id", "smoke", "fitspatrick", "region", "diagnostic", "img_id", "biopsed")


def _write(path: Path, rows: list[dict], header=HEADER) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(i: int, diagnostic: str = "BCC", fst: str = "2", lesion: int | None = None,
         patient: int | None = None) -> dict:
    lesion = i if lesion is None else lesion
    patient = i if patient is None else patient
    return {"patient_id": f"PAT_{patient}", "lesion_id": str(lesion), "fitspatrick": fst,
            "diagnostic": diagnostic, "img_id": f"PAT_{patient}_{lesion}_{i}.png", "biopsed": "True"}


def _published_like() -> list[dict]:
    """2298 rows with the article's counts: 1641 lesions, 1373 patients."""
    diagnostics = [dx for dx, n in pu.EXPECTED_DIAGNOSTIC_COUNTS.items() for _ in range(n)]
    return [_row(i, dx, lesion=i % 1641, patient=(i % 1641) % 1373) for i, dx in enumerate(diagnostics)]


class TestFitzpatrick:
    @pytest.mark.parametrize("raw, want", [
        ("1", (1, "ok")), ("6", (6, "ok")), ("3.0", (3, "ok")), (" 2 ", (2, "ok")),
        ("", (None, "missing")), ("  ", (None, "missing")), (None, (None, "missing")),
        ("UNK", (None, "missing")), ("nan", (None, "missing")),
        ("0", (None, "out_of_range")), ("7", (None, "out_of_range")),
        ("2.5", (None, "out_of_range")), ("IV", (None, "out_of_range")),
    ])
    def test_values_blanks_and_junk(self, raw, want):
        assert pu.parse_fst(raw) == want

    @pytest.mark.parametrize("value, want", [
        (1, "I"), (2, "II"), (3, "III-VI"), (4, "III-VI"), (6, "III-VI"), (None, pu.FST_MISSING),
    ])
    def test_the_registered_bins(self, value, want):
        assert pu.fst_bin(value) == want

    def test_missing_values_survive_parsing_and_land_in_their_own_bin(self, tmp_path):
        rows = [_row(0, fst="1"), _row(1, fst=""), _row(2, fst="UNK"), _row(3, fst="9"), _row(4, fst="5")]
        records = pu.parse_metadata(_write(tmp_path / "m.csv", rows))
        summary = pu.fst_summary(records)
        assert summary["by_bin"] == {"I": 1, "II": 0, "III-VI": 1, pu.FST_MISSING: 3}
        assert summary["by_status"] == {"ok": 2, "missing": 2, "out_of_range": 1}


class TestParsing:
    def test_labels_map_to_ham_as_registered(self, tmp_path):
        rows = [_row(i, dx) for i, dx in enumerate(("ACK", "BCC", "MEL", "NEV", "SCC", "SEK"))]
        records = pu.parse_metadata(_write(tmp_path / "m.csv", rows))
        assert [r.ham_label for r in records] == ["akiec", "bcc", "mel", "nv", "akiec", "bkl"]

    def test_img_id_is_keyed_without_the_extension(self, tmp_path):
        records = pu.parse_metadata(_write(tmp_path / "m.csv", [_row(7)]))
        assert records[0].img_id == "PAT_7_7_7"
        assert pu.image_key("PAT_7_7_7") == pu.image_key("PAT_7_7_7.PNG") == "PAT_7_7_7"

    def test_a_missing_column_is_refused(self, tmp_path):
        header = tuple(c for c in HEADER if c != "fitspatrick")
        with pytest.raises(pu.PadMetadataError, match="fitspatrick"):
            pu.parse_metadata(_write(tmp_path / "m.csv", [_row(0)], header=header))

    def test_an_unknown_diagnostic_is_refused(self, tmp_path):
        with pytest.raises(pu.PadMetadataError, match="BOD"):
            pu.parse_metadata(_write(tmp_path / "m.csv", [_row(0, "BOD")]))

    def test_a_blank_patient_is_refused(self, tmp_path):
        row = _row(0)
        row["patient_id"] = ""
        with pytest.raises(pu.PadMetadataError, match="patient_id"):
            pu.parse_metadata(_write(tmp_path / "m.csv", [row]))

    def test_a_repeated_image_is_refused(self, tmp_path):
        with pytest.raises(pu.PadMetadataError, match="repetidos"):
            pu.parse_metadata(_write(tmp_path / "m.csv", [_row(0), _row(0)]))


class TestPublishedCounts:
    def test_the_published_shape_passes(self, tmp_path):
        records = pu.parse_metadata(_write(tmp_path / "m.csv", _published_like()))
        assert pu.count_discrepancies(records) == []
        pu.check_published_counts(records)

    def test_one_image_short_is_refused(self, tmp_path):
        records = pu.parse_metadata(_write(tmp_path / "m.csv", _published_like()[:-1]))
        with pytest.raises(pu.PadMetadataError, match="imagenes: 2297"):
            pu.check_published_counts(records)

    def test_a_lesion_shared_by_two_patients_is_reported(self, tmp_path):
        rows = [_row(0, lesion=5, patient=1), _row(1, lesion=5, patient=2)]
        problems = pu.count_discrepancies(pu.parse_metadata(_write(tmp_path / "m.csv", rows)))
        assert any("mas de un paciente" in p for p in problems)

    def test_readings_keep_only_mapped_classes(self):
        mapped = set(pu.PAD_TO_HAM.values())
        for name, classes in pu.READINGS.items():
            assert set(classes) <= mapped, name
            assert list(classes) == sorted(classes), "el orden de las columnas es el de HAM (alfabetico)"
        assert "bkl" not in pu.READINGS["primary"] + pu.READINGS["secondary"]
