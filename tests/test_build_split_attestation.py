"""A declared gap, not a filled one.

Two arms of this run have incomplete manifests. The temptation is to write
the right value into the run.json afterwards, which would make the file
indistinguishable from one that was correct during training. These tests pin
the alternative: the gaps are detected and reported, the later check is
recorded with its limits, and no manifest is touched.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_split_attestation as att  # noqa: E402

POOL_HASH = "70846dbcbf60c3b1c694a40485b720bcf980e204e6aa5bef5694dba73329a67c"
POOL_COUNT = 55034


def _run(**overrides) -> dict:
    payload = {
        "split_hash": "a" * 64,
        "code_revision": "301046f",
        "data_revision": "pool-final",
        "execution": {"dataset_image_count": POOL_COUNT},
    }
    payload.update(overrides)
    return payload


def _tone(tmp_path: Path, arms: dict[str, dict]) -> Path:
    tone = tmp_path / "tone"
    for name, run in arms.items():
        directory = tone / name
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return tone


def test_a_zeroed_split_hash_is_reported_as_a_gap():
    gaps = att.gaps_for("ssl_plain_vits14", _run(split_hash=att.ZERO_HASH))
    assert any("ceros" in gap for gap in gaps)


def test_a_null_code_revision_is_reported_as_a_gap():
    assert any("code_revision" in gap for gap in att.gaps_for("x", _run(code_revision=None)))


def test_a_complete_manifest_has_no_gaps():
    assert att.gaps_for("ssl_plain_vitb14", _run()) == []


def test_the_image_count_corroborates_the_pool():
    arms = {"ssl_plain_vits14": _run(split_hash=att.ZERO_HASH)}
    result = att.build(arms, POOL_HASH, POOL_COUNT)
    assert result["brazos"]["ssl_plain_vits14"]["coincide_con_el_pool_actual"] is True


def test_a_different_image_count_is_flagged():
    arms = {"ssl_plain_vits14": _run(execution={"dataset_image_count": 40000})}
    result = att.build(arms, POOL_HASH, POOL_COUNT)
    assert result["brazos"]["ssl_plain_vits14"]["coincide_con_el_pool_actual"] is False


def test_a_missing_count_is_not_reported_as_a_match():
    arms = {"ssl_plain_vits14": _run(execution={})}
    result = att.build(arms, POOL_HASH, POOL_COUNT)
    assert result["brazos"]["ssl_plain_vits14"]["coincide_con_el_pool_actual"] is None


def test_the_attestation_states_what_it_does_not_prove():
    result = att.build({"a": _run()}, POOL_HASH, POOL_COUNT)
    disclaimer = result["que_NO_prueba"]
    assert "evidencia, no demostracion" in disclaimer, "no debe leerse como demostracion"
    assert "no se rellenan" in disclaimer, "debe decir que los huecos no se rellenaron"
    assert "identico durante el entrenamiento" in disclaimer


def test_the_manifests_are_never_modified(tmp_path):
    arms = {"ssl_plain_vits14": _run(split_hash=att.ZERO_HASH)}
    tone = _tone(tmp_path, arms)
    manifest = tone / "ssl_plain_vits14" / "run.json"
    before = manifest.read_bytes()

    att.build(att.read_arms(tone), POOL_HASH, POOL_COUNT)

    assert manifest.read_bytes() == before, "la atestacion no debe tocar la evidencia"


def test_every_arm_with_a_manifest_is_read(tmp_path):
    tone = _tone(tmp_path, {
        "ssl_plain_vits14": _run(split_hash=att.ZERO_HASH),
        "ssl_proposed_vits14": _run(code_revision=None),
        "ssl_plain_vitb14": _run(),
    })
    assert set(att.read_arms(tone)) == {
        "ssl_plain_vits14", "ssl_proposed_vits14", "ssl_plain_vitb14"
    }


def test_the_markdown_marks_the_gap_per_arm():
    arms = {
        "ssl_plain_vits14": _run(split_hash=att.ZERO_HASH),
        "ssl_plain_vitb14": _run(),
    }
    text = att.to_markdown(att.build(arms, POOL_HASH, POOL_COUNT))
    assert "ceros" in text
    assert "| `ssl_plain_vitb14` | ninguno |" in text
    assert "Que NO prueba" in text


def test_an_empty_tone_directory_is_refused(tmp_path, monkeypatch):
    tone = tmp_path / "tone"
    tone.mkdir()
    receipt = tmp_path / "pool.json"
    receipt.write_text(json.dumps({"split_hash": POOL_HASH}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "build_split_attestation.py", "--tone", str(tone),
        "--pool-receipt", str(receipt), "--pool-count", str(POOL_COUNT),
        "--out", str(tmp_path / "out.json"),
    ])
    with pytest.raises(SystemExit, match="run.json"):
        att.main()
