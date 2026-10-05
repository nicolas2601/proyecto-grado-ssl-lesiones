"""Path resolution is tested against fake directory trees - no network, no Kaggle."""

import pytest

from dermssl import data_resolver as dr


def make_ham(root, n_images=3):
    """Minimal HAM10000 layout: the metadata CSV plus some jpgs beside it."""
    ham = root / "skin-cancer-mnist-ham10000"
    ham.mkdir(parents=True, exist_ok=True)
    (ham / "HAM10000_metadata.csv").write_text("lesion_id,image_id,dx\n", encoding="utf-8")
    images = ham / "HAM10000_images_part_1"
    images.mkdir(exist_ok=True)
    for i in range(n_images):
        (images / f"ISIC_{i:07d}.jpg").write_bytes(b"fake")
    return ham


def make_pool(root, name="isic2020", n_images=2):
    pool = root / name
    pool.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        (pool / f"ISIC_9{i:06d}.jpg").write_bytes(b"fake")
    return pool


def test_detect_backend_prefers_mounted_kaggle_input(tmp_path):
    kaggle = tmp_path / "kaggle_input"
    make_ham(kaggle)
    assert dr.detect_backend(kaggle_input=kaggle) == "kaggle-input"


def test_detect_backend_falls_back_to_local_then_kagglehub(tmp_path):
    empty_kaggle = tmp_path / "nope"
    local = tmp_path / "scratch"
    make_ham(local)
    assert dr.detect_backend(kaggle_input=empty_kaggle, local_root=local) == "local"
    assert dr.detect_backend(kaggle_input=empty_kaggle, local_root=None) == "kagglehub"


def test_detect_backend_ignores_empty_kaggle_input(tmp_path):
    empty = tmp_path / "kaggle_input"
    empty.mkdir()
    assert dr.detect_backend(kaggle_input=empty) == "kagglehub"


def test_local_backend_resolves_ham_and_pool(tmp_path):
    make_ham(tmp_path, n_images=3)
    make_pool(tmp_path, n_images=2)
    data = dr.resolve_datasets(backend="local", local_root=tmp_path)

    assert data.backend == "local"
    assert data.ham_metadata_csv.name == "HAM10000_metadata.csv"
    assert len(data.ham_images) == 3
    assert len(data.pool_images) == 2, "pool must exclude the labelled HAM images"
    assert set(data.ham_images).isdisjoint(set(data.pool_images))


def test_summary_is_json_friendly(tmp_path):
    make_ham(tmp_path)
    summary = dr.resolve_datasets(backend="local", local_root=tmp_path).summary()
    assert summary["backend"] == "local"
    assert summary["ham_images"] == 3
    assert isinstance(summary["roots"][0], str)


def test_skipping_the_pool_index_avoids_the_expensive_walk(tmp_path):
    """On Kaggle, indexing 48 442 pool images cost 157 s of a 192 s smoke run."""
    make_ham(tmp_path, n_images=3)
    make_pool(tmp_path, n_images=5)

    data = dr.resolve_datasets(backend="local", local_root=tmp_path, index_pool=False)
    assert data.pool_images == {}
    assert data.pool_indexed is False
    assert len(data.ham_images) == 3, "the labelled set is still indexed"
    # An empty dict must not be reported as "we looked and found zero images".
    assert data.summary()["pool_images"] == "no indexado"


def test_indexing_the_pool_is_still_the_default(tmp_path):
    make_ham(tmp_path, n_images=3)
    make_pool(tmp_path, n_images=5)
    data = dr.resolve_datasets(backend="local", local_root=tmp_path)
    assert data.pool_indexed is True
    assert len(data.pool_images) == 5
    assert data.summary()["pool_images"] == 5


def test_skipping_the_pool_does_not_hide_a_missing_ham(tmp_path):
    make_pool(tmp_path)
    with pytest.raises(dr.DataResolutionError):
        dr.resolve_datasets(backend="local", local_root=tmp_path, index_pool=False)


def test_missing_ham_raises_an_actionable_error(tmp_path):
    make_pool(tmp_path)  # pool present, HAM absent
    with pytest.raises(dr.DataResolutionError) as err:
        dr.resolve_datasets(backend="local", local_root=tmp_path)
    assert "HAM10000_metadata.csv" in str(err.value)


def test_local_backend_requires_a_root():
    with pytest.raises(dr.DataResolutionError, match="local_root"):
        dr.dataset_roots("local", ["ham10000"], local_root=None)


def test_nonexistent_local_root_raises(tmp_path):
    with pytest.raises(dr.DataResolutionError, match="does not exist"):
        dr.dataset_roots("local", ["ham10000"], local_root=tmp_path / "ghost")


def test_kaggle_input_backend_requires_the_mount(tmp_path):
    with pytest.raises(dr.DataResolutionError, match="inside Kaggle"):
        dr.dataset_roots("kaggle-input", ["ham10000"], kaggle_input=tmp_path / "ghost")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        dr.dataset_roots("telepathy", ["ham10000"])


def test_kagglehub_backend_fails_early_without_credentials(monkeypatch):
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(dr.Path, "home", staticmethod(lambda: dr.Path("/nonexistent-home")))
    assert dr.credentials_present() is False
    with pytest.raises(dr.DataResolutionError, match="credentials"):
        dr.dataset_roots("kagglehub", ["ham10000"])


def test_credentials_detected_from_environment(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "nicolas")
    monkeypatch.setenv("KAGGLE_KEY", "secret")
    assert dr.credentials_present() is True


def test_index_images_later_root_wins_on_collision(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "ISIC_0000001.jpg").write_bytes(b"x")
    resolved = dr.index_images([a, b])
    assert resolved["ISIC_0000001"].parent == b


def test_index_images_skips_missing_roots(tmp_path):
    assert dr.index_images([tmp_path / "ghost"]) == {}


def test_find_label_csvs_is_case_insensitive(tmp_path):
    (tmp_path / "ISIC_2019_Training_GroundTruth.csv").write_text("MEL,NV\n", encoding="utf-8")
    (tmp_path / "unrelated.csv").write_text("a\n", encoding="utf-8")
    hits = dr.find_label_csvs([tmp_path])
    assert [p.name for p in hits] == ["ISIC_2019_Training_GroundTruth.csv"]


def test_dataset_handles_cover_the_three_datasets():
    assert set(dr.DATASET_HANDLES) == {"ham10000", "isic2019", "isic2020"}


def test_build_pool_manifest_preserves_source_identity_and_reports_coverage(tmp_path):
    ham = make_ham(tmp_path, n_images=1)
    isic2019 = make_pool(tmp_path, name="isic2019", n_images=1)
    isic2020 = make_pool(tmp_path, name="isic2020", n_images=1)

    manifest = dr.build_pool_manifest(
        {
            "ham10000": [next((ham / "HAM10000_images_part_1").glob("*.jpg"))],
            "isic2019": [next(isic2019.glob("*.jpg"))],
            "isic2020": [next(isic2020.glob("*.jpg"))],
        },
        source_revisions={
            "ham10000": "ham-v1",
            "isic2019": "isic2019-resized-v1",
            "isic2020": "isic2020-resized-v1",
        },
        expected_counts={"ham10000": 1, "isic2019": 1, "isic2020": 1},
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["total_images"] == 3
    assert manifest["coverage"]["complete"] is True
    assert {(row["dataset"], row["image_id"], row["source_revision"]) for row in manifest["images"]} == {
        ("ham10000", "ISIC_0000000", "ham-v1"),
        ("isic2019", "ISIC_9000000", "isic2019-resized-v1"),
        ("isic2020", "ISIC_9000000", "isic2020-resized-v1"),
    }
    assert all(len(row["sha256"]) == 64 for row in manifest["images"])
