"""The B* seed of the E3 robustness run: weights folder, feature key, typed refusal, fingerprint."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e3_io  # noqa: E402
import v71_io  # noqa: E402

from dermssl import e3_seed as es  # noqa: E402
from dermssl import e3_store as st  # noqa: E402
from dermssl import v7_seeds  # noqa: E402


def make_weights(root: Path, seed: int, payload: bytes) -> Path:
    path = root / es.weights_dir_name(seed) / es.WEIGHTS_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize("seed, folder", [(42, "final_B"), (43, "final_B_s43"), (44, "final_B_s44")])
def test_the_weights_folder_follows_the_v7_layout(seed, folder):
    assert es.weights_dir_name(seed) == folder == v7_seeds.final_dir_name("B", seed)


def test_the_weights_file_name_is_the_one_the_bank_reads():
    assert es.WEIGHTS_FILE == v71_io.WEIGHTS_FILE


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_feature_key_and_weights_name_carry_the_seed(seed):
    assert es.weights_name(seed) == f"B_s{seed}_a1.0"
    assert es.feature_key(seed) == f"B_s{seed}_a1.0@224|cls4"
    assert es.protocol(seed) == f"B_s{seed}_a1.0@224|cls4|balanced|tta"


def test_the_base_seed_protocol_is_the_registered_one():
    assert es.protocol(42) == e3_io.PROTOCOL
    assert es.weights_name(42) == e3_io.B71_WEIGHTS


@pytest.mark.parametrize("seed", [0, 41, 45, -1])
def test_an_unregistered_seed_is_refused(seed):
    with pytest.raises(es.UnknownSeedError):
        es.weights_name(seed)


def test_weights_path_returns_the_existing_file(tmp_path):
    path = make_weights(tmp_path, 43, b"w43")
    assert es.weights_path(tmp_path, 43) == path


def test_missing_weights_raise_a_typed_error_that_wraps_the_os_error(tmp_path):
    make_weights(tmp_path, 42, b"w42")
    with pytest.raises(es.MissingWeightsError) as info:
        es.weights_path(tmp_path, 44)
    assert isinstance(info.value, st.E3Error)
    assert isinstance(info.value.__cause__, OSError)
    assert "44" in str(info.value) and "final_B_s44" in str(info.value)


def test_a_folder_in_place_of_the_weights_file_is_missing_weights(tmp_path):
    (tmp_path / "final_B_s43" / es.WEIGHTS_FILE).mkdir(parents=True)
    with pytest.raises(es.MissingWeightsError):
        es.weights_path(tmp_path, 43)


def test_the_base_seed_fingerprint_adds_no_new_key(tmp_path):
    make_weights(tmp_path, 42, b"w42")
    fp = es.fingerprint(tmp_path, 42)
    assert set(fp) == {"protocol", "weights_sha256"} and fp["protocol"] == e3_io.PROTOCOL


def test_another_seed_records_seed_protocol_and_weights_sha(tmp_path):
    make_weights(tmp_path, 42, b"w42")
    make_weights(tmp_path, 43, b"w43")
    base, other = es.fingerprint(tmp_path, 42), es.fingerprint(tmp_path, 43)
    assert other["b71_seed"] == 43 and other["protocol"] == es.protocol(43)
    assert other["weights_sha256"] != base["weights_sha256"]
    assert st.canonical_sha256(base) != st.canonical_sha256(other)
