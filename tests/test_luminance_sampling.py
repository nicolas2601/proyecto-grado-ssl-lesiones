"""T7 (docs/27): luminance terciles per cache row and the fixed-budget oversampled epoch."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

import seg_paths  # noqa: F401
from dermssl import luminance_sampling as ls
from sampling import shard_batches

REPO = Path(__file__).resolve().parents[1]
FST_COLUMNS = ["image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita", "n_skin_px"]


def _manifest(path: Path, rows: list[tuple[str, str]]) -> str:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FST_COLUMNS)
        writer.writerows([i, "", t, "ok", "70", "10", "40", "100"] for i, t in rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_real_registration_precedes_the_t7_training_code():
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True).stdout.strip()
    if shallow != "false":
        pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
    assert ls.check_registration(REPO) == ls.REGISTRATION_COMMIT


def test_terciles_follow_the_cache_rows_and_missing_ids_are_unavailable(tmp_path):
    digest = _manifest(tmp_path / "m.csv", [("b", "dark"), ("a", "light"), ("c", "unavailable")])
    rows = [{"index": 0, "image_id": "a"}, {"index": 1, "image_id": "b"}, {"index": 2, "image_id": "z"},
            {"index": 3, "image_id": "c"}]
    labels = ls.terciles_by_row(rows, tmp_path / "m.csv", digest)
    assert labels.tolist() == ["light", "dark", "unavailable", "unavailable"]
    with pytest.raises(ValueError, match="sha256"):
        ls.terciles_by_row(rows, tmp_path / "m.csv", "0" * 64)


def test_only_medium_and_light_are_eligible_for_darkening():
    labels = np.array(["dark", "medium", "light", "unavailable"])
    assert ls.darken_eligible(labels).tolist() == [False, True, True, False]


@pytest.mark.parametrize("epoch", [0, 3])
@pytest.mark.parametrize("rank", [0, 1])
def test_k1_reproduces_a2s_batches_exactly(epoch, rank):
    train = np.arange(5, 5 + 203) * 3
    is_dark = np.arange(203) % 3 == 0
    plan = ls.oversampled_epoch(train, is_dark, k=1, n_samples=train.size, seed=42, epoch=epoch)
    t7 = shard_batches(plan, 4, rank, 2, epoch, 42, shuffle=False)
    a2 = shard_batches(train, 4, rank, 2, epoch, 42)
    assert len(t7) == len(a2) and all(np.array_equal(x, y) for x, y in zip(t7, a2))


def test_k2_keeps_the_budget_and_raises_the_dark_share():
    train = np.arange(9000)
    is_dark = np.arange(9000) % 3 == 0
    plan = ls.oversampled_epoch(train, is_dark, k=2, n_samples=train.size, seed=42, epoch=0)
    assert plan.size == train.size
    share = np.isin(plan, train[is_dark]).mean()
    assert share == pytest.approx(2 * 3000 / 12000, abs=0.01)
    assert np.bincount(plan).max() <= 2
    assert not np.array_equal(plan, ls.oversampled_epoch(train, is_dark, 2, train.size, 42, 1))
    assert np.array_equal(plan, ls.oversampled_epoch(train, is_dark, 2, train.size, 42, 0))


def test_invalid_factors_and_budgets_are_refused():
    train, is_dark = np.arange(10), np.zeros(10, bool)
    with pytest.raises(ValueError, match="k"):
        ls.oversampled_epoch(train, is_dark, k=0, n_samples=10, seed=0, epoch=0)
    with pytest.raises(ValueError, match="muestras"):
        ls.oversampled_epoch(train, is_dark, k=1, n_samples=11, seed=0, epoch=0)
    with pytest.raises(ValueError, match="longitud"):
        ls.oversampled_epoch(train, is_dark[:9], k=1, n_samples=10, seed=0, epoch=0)


def _teacher_repo(tmp_path: Path, addendum: bool, marker: bool = False, teacher_dark: float = 0.78,
                  **provenance) -> Path:
    """A fake repo: rows where r0's dark tercile scores ``teacher_dark`` against A2's 0.70 (rest 0.80 for both)."""
    import json

    result = tmp_path / ls.TEACHER_CHECK_DIR
    result.mkdir(parents=True)
    ids = [("d1", "dark"), ("d2", "dark"), ("m1", "medium"), ("l1", "light"), ("x1", "dark")]
    with open(result / ls.TEACHER_CHECK_ROWS, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "tone_tercile", "excluded", "dice_binary", "dice_confident"])
        for image_id, tercile in ids:
            if image_id == "x1":
                writer.writerow([image_id, tercile, "True", "", ""])
                continue
            value = teacher_dark if tercile == "dark" else 0.80
            writer.writerow([image_id, tercile, "False", value, value])
    a2 = tmp_path / ls.A2_HAM_PER_IMAGE
    a2.parent.mkdir(parents=True)
    with open(a2, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "dice"])
        writer.writerows([i, 0.70 if t == "dark" else 0.80] for i, t in ids)
    rows_sha = hashlib.sha256((result / ls.TEACHER_CHECK_ROWS).read_bytes()).hexdigest()
    prov = {"registration_commit": ls.REGISTRATION_COMMIT, "manifest_sha256": ls.MANIFEST_SHA256,
            "pseudo_r0_sha256": ls.PSEUDO_R0_SHA256, "cache_manifest_sha256": ls.CACHE_MANIFEST_SHA256,
            "rows_sha256": rows_sha, **provenance}
    doc = {"analysis": {"addendum_required": addendum}, "provenance": prov}
    (result / ls.TEACHER_CHECK_JSON).write_text(json.dumps(doc), encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / Path(ls.REGISTRATION_DOC).name).write_text(ls.ADDENDUM_MARKER if marker else "doc", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fake_a2_sha(monkeypatch):
    def pin(repo: Path) -> Path:
        monkeypatch.setattr(ls, "A2_HAM_PER_IMAGE_SHA256",
                            hashlib.sha256((repo / ls.A2_HAM_PER_IMAGE).read_bytes()).hexdigest())
        return repo
    return pin


def test_the_committed_teacher_check_opens_the_gate():
    gate = ls.check_teacher_gate(REPO)
    assert gate["addendum_required"] is False and len(gate["rows_sha256"]) == 64


def test_no_teacher_check_means_no_training(tmp_path):
    with pytest.raises(ls.TeacherGateError, match="chequeo del maestro"):
        ls.check_teacher_gate(tmp_path)


@pytest.mark.parametrize("key", ["registration_commit", "manifest_sha256", "pseudo_r0_sha256",
                                 "cache_manifest_sha256", "rows_sha256"])
def test_a_teacher_check_on_other_inputs_is_refused(tmp_path, key, fake_a2_sha):
    with pytest.raises(ls.TeacherGateError, match=key):
        ls.check_teacher_gate(fake_a2_sha(_teacher_repo(tmp_path, False, **{key: "0" * 64})))


def test_a_triggered_addendum_must_be_committed_before_training(tmp_path, fake_a2_sha):
    with pytest.raises(ls.TeacherGateError, match="adenda"):
        ls.check_teacher_gate(fake_a2_sha(_teacher_repo(tmp_path / "a", True, teacher_dark=0.60)))
    repo = fake_a2_sha(_teacher_repo(tmp_path / "b", True, marker=True, teacher_dark=0.60))
    assert ls.check_teacher_gate(repo)["addendum_required"] is True


@pytest.mark.parametrize(("flag", "teacher_dark"), [(False, 0.60), (True, 0.78)])
def test_the_addendum_flag_must_match_the_one_recomputed_from_the_hashed_rows(tmp_path, fake_a2_sha, flag,
                                                                             teacher_dark):
    repo = fake_a2_sha(_teacher_repo(tmp_path, flag, marker=True, teacher_dark=teacher_dark))
    with pytest.raises(ls.TeacherGateError, match="recalculad"):
        ls.check_teacher_gate(repo)


def test_a2s_reference_must_be_the_published_one(tmp_path):
    with pytest.raises(ls.TeacherGateError, match="A2"):
        ls.check_teacher_gate(_teacher_repo(tmp_path, False))
