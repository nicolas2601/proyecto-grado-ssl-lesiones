"""PAD-UFES-20 patient split into target_adapt / target_test (docs/33 section 4)."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from dermssl import pad_target_split as ts

DIAGNOSTICS = ("BCC", "ACK", "NEV", "MEL")


def _rows(n_patients: int = 40, per_patient: int = 3) -> list[dict]:
    rows = []
    for p in range(n_patients):
        dx = DIAGNOSTICS[p % len(DIAGNOSTICS)]
        for i in range(per_patient):
            rows.append({"img_id": f"PAT_{p}_{i}_{p * 10 + i}", "patient_id": f"PAT_{p}",
                         "lesion_id": str(i // 2), "diagnostic": dx})
    return rows


def _unique_hash(row: dict) -> str:
    return "h-" + row["img_id"]


class TestDedup:
    def test_unique_rows_pass_through(self):
        kept, report = ts.dedup(_rows(4), _unique_hash)
        assert len(kept) == 12 and report["n_kept"] == 12 and report["n_input"] == 12

    def test_a_repeated_img_id_is_dropped_once(self):
        rows = _rows(2)
        kept, report = ts.dedup(rows + [dict(rows[0])], _unique_hash)
        assert len(kept) == len(rows)
        assert report["dropped_duplicate_img_id"] == [rows[0]["img_id"]]

    def test_identical_bytes_under_another_id_are_dropped(self):
        rows = _rows(2)
        same = lambda row: "same" if row["img_id"] in {rows[0]["img_id"], rows[3]["img_id"]} else _unique_hash(row)  # noqa: E731
        kept, report = ts.dedup(rows, same)
        dropped = report["dropped_duplicate_content"]
        assert len(kept) == len(rows) - 1 and len(dropped) == 1
        assert dropped[0] == {"img_id": rows[3]["img_id"], "same_as": rows[0]["img_id"]}

    def test_the_kept_copy_does_not_depend_on_input_order(self):
        rows = _rows(2)
        same = lambda row: "same" if row["img_id"] in {rows[0]["img_id"], rows[3]["img_id"]} else _unique_hash(row)  # noqa: E731
        forward, _ = ts.dedup(rows, same)
        backward, _ = ts.dedup(rows[::-1], same)
        assert {r["img_id"] for r in forward} == {r["img_id"] for r in backward}


class TestSplitByPatient:
    def test_every_patient_is_on_exactly_one_side(self):
        adapt, test = ts.split_by_patient(_rows())
        assert not {r["patient_id"] for r in adapt} & {r["patient_id"] for r in test}

    def test_every_row_lands_on_one_side(self):
        rows = _rows()
        adapt, test = ts.split_by_patient(rows)
        assert sorted(r["img_id"] for r in adapt + test) == sorted(r["img_id"] for r in rows)

    def test_no_lesion_key_spans_both_sides(self):
        adapt, test = ts.split_by_patient(_rows())
        key = lambda r: (r["patient_id"], r["lesion_id"])  # noqa: E731
        assert not {key(r) for r in adapt} & {key(r) for r in test}

    def test_the_proportion_of_patients_is_the_requested_one(self):
        adapt, test = ts.split_by_patient(_rows(40))
        assert len({r["patient_id"] for r in adapt}) == 20
        assert len({r["patient_id"] for r in test}) == 20

    def test_each_diagnostic_is_split_close_to_the_fraction(self):
        adapt, test = ts.split_by_patient(_rows(40))
        a, t = Counter(r["diagnostic"] for r in adapt), Counter(r["diagnostic"] for r in test)
        for dx in DIAGNOSTICS:
            assert abs(a[dx] - t[dx]) <= 3 * 2  # one patient (3 images) each way

    def test_it_is_deterministic_for_a_seed_and_changes_with_it(self):
        ids = lambda split: sorted(r["img_id"] for r in split[0])  # noqa: E731
        assert ids(ts.split_by_patient(_rows())) == ids(ts.split_by_patient(_rows()))
        assert ids(ts.split_by_patient(_rows(), seed=1)) != ids(ts.split_by_patient(_rows(), seed=2))

    def test_the_input_order_does_not_matter(self):
        rows = _rows()
        ids = lambda split: sorted(r["img_id"] for r in split[0])  # noqa: E731
        assert ids(ts.split_by_patient(rows)) == ids(ts.split_by_patient(rows[::-1]))

    def test_a_rare_diagnostic_with_few_patients_still_reaches_both_sides(self):
        rows = _rows(40) + [{"img_id": f"PAT_r{p}_0_0", "patient_id": f"PAT_r{p}", "lesion_id": "0",
                              "diagnostic": "SCC"} for p in range(4)]
        adapt, test = ts.split_by_patient(rows)
        assert {r["diagnostic"] for r in adapt} >= {"SCC"} and {r["diagnostic"] for r in test} >= {"SCC"}

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_a_fraction_outside_the_open_unit_interval_is_refused(self, fraction):
        with pytest.raises(ts.TargetSplitError, match="fraction"):
            ts.split_by_patient(_rows(), adapt_fraction=fraction)

    def test_one_patient_cannot_be_split(self):
        with pytest.raises(ts.TargetSplitError, match="patients"):
            ts.split_by_patient(_rows(1))


class TestDocument:
    def _doc(self):
        return ts.build_document(_rows(), _unique_hash, manifest_sha256="m" * 64)

    def test_it_records_both_lists_counts_and_their_hashes(self):
        doc = self._doc()
        for side in ("target_adapt", "target_test"):
            ids = doc[side]["img_ids"]
            assert doc[side]["n_images"] == len(ids) == sum(doc[side]["by_diagnostic"].values())
            assert doc["sha256"][side] == ts.ids_sha256(ids)
        assert doc["seed"] == ts.SEED and doc["adapt_fraction"] == 0.5
        assert doc["manifest_sha256"] == "m" * 64

    def test_the_two_lists_share_no_image_patient_or_content(self):
        doc = self._doc()
        assert not set(doc["target_adapt"]["img_ids"]) & set(doc["target_test"]["img_ids"])
        assert not set(doc["target_adapt"]["patient_ids"]) & set(doc["target_test"]["patient_ids"])

    def test_the_hash_does_not_depend_on_list_order(self):
        assert ts.ids_sha256(["b", "a"]) == ts.ids_sha256(["a", "b"])

    def test_a_content_collision_across_the_sides_is_refused(self):
        adapt = [{"img_id": "a", "patient_id": "P1", "lesion_id": "0", "diagnostic": "BCC"}]
        test = [{"img_id": "b", "patient_id": "P2", "lesion_id": "0", "diagnostic": "BCC"}]
        with pytest.raises(ts.TargetSplitError, match="content"):
            ts.check_disjoint(adapt, test, lambda row: "same")


class TestTargetTestLock:
    def _write(self, tmp_path):
        path = tmp_path / "split.json"
        path.write_text(json.dumps(ts.build_document(_rows(), _unique_hash, manifest_sha256="m" * 64)))
        return path

    def test_the_first_read_returns_the_ids_and_creates_the_lock(self, tmp_path):
        path = self._write(tmp_path)
        lock = ts.TargetTestLock(tmp_path)
        ids = ts.read_target_test_ids(path, lock)
        assert ids == json.loads(path.read_text())["target_test"]["img_ids"]
        assert (tmp_path / ts.LOCK_NAME).is_file()

    def test_a_second_read_refuses(self, tmp_path):
        path = self._write(tmp_path)
        ts.read_target_test_ids(path, ts.TargetTestLock(tmp_path))
        with pytest.raises(ts.TargetTestAlreadyReadError, match="target_test"):
            ts.read_target_test_ids(path, ts.TargetTestLock(tmp_path))

    def test_a_tampered_list_refuses_without_burning_the_lock(self, tmp_path):
        path = self._write(tmp_path)
        doc = json.loads(path.read_text())
        doc["target_test"]["img_ids"].append("PAT_999_0_0")
        path.write_text(json.dumps(doc))
        with pytest.raises(ts.TargetTestPinError):
            ts.read_target_test_ids(path, ts.TargetTestLock(tmp_path))
        assert not (tmp_path / ts.LOCK_NAME).exists()

    def test_the_adapt_list_is_free_to_read_and_pinned(self, tmp_path):
        path = self._write(tmp_path)
        assert ts.read_target_adapt_ids(path) == json.loads(path.read_text())["target_adapt"]["img_ids"]
        assert not (tmp_path / ts.LOCK_NAME).exists()

    def test_the_lock_records_the_list_hash_and_the_time(self, tmp_path):
        path = self._write(tmp_path)
        lock = ts.TargetTestLock(tmp_path, clock=lambda: "2026-10-05T00:00:00+00:00")
        ts.read_target_test_ids(path, lock)
        body = json.loads((tmp_path / ts.LOCK_NAME).read_text())
        assert body["read_at"] == "2026-10-05T00:00:00+00:00"
        assert body["target_test_sha256"] == json.loads(path.read_text())["sha256"]["target_test"]
