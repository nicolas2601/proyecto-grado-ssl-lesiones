"""Side-by-side tone/FST distributions of two FST manifests (docs/27, FST sin pelo)."""

import json

import pytest

import seg_paths  # noqa: F401
import compare_fst_manifests as cmp


def _dataset(tone, fst, flagged=None):
    summary = {"n": sum(tone.values()), "tone_distribution": tone, "FST_distribution": fst}
    if flagged is not None:
        summary["flagged"] = flagged
    return summary


def _flagged(n, artifact, low):
    return {"n": n, "n_artifact_detected": artifact, "n_confidence_below_1": low}


def _manifest(by_dataset):
    return {"schema_version": "1.1", "records": [], "summary": {"by_dataset": by_dataset}}


ORIGINAL = _manifest({
    "ham": _dataset({"dark": 2, "medium": 1, "light": 1}, {"1": 3, "2": 1}, _flagged(2, 1, 2)),
    "isic": _dataset({"light": 4}, {"1": 4}, _flagged(1, 0, 1)),
})
HAIR_FREE = _manifest({
    "ham": _dataset({"dark": 1, "medium": 2, "light": 1}, {"1": 2, "2": 1, "6": 1}, _flagged(1, 0, 1)),
    "isic": _dataset({"light": 4}, {"1": 4}, _flagged(0, 0, 0)),
})
HAM_ONLY = _manifest({"ham": HAIR_FREE["summary"]["by_dataset"]["ham"]})


def test_each_tone_bin_carries_counts_proportions_and_the_difference():
    result = cmp.compare(ORIGINAL, HAIR_FREE)
    dark = result["scopes"]["ham"]["tone_distribution"]["bins"]["dark"]
    assert dark == {
        "a_count": 2, "b_count": 1, "a_prop": 0.5, "b_prop": 0.25,
        "diff_count": -1, "diff_prop": -0.25,
    }
    assert result["scopes"]["ham"]["tone_distribution"]["total"] == {"a": 4, "b": 4}


def test_a_bin_present_in_one_manifest_counts_as_zero_in_the_other():
    fst = cmp.compare(ORIGINAL, HAIR_FREE)["scopes"]["ham"]["FST_distribution"]["bins"]
    assert list(fst) == ["1", "2", "6"]
    assert fst["6"]["a_count"] == 0 and fst["6"]["b_count"] == 1
    assert fst["6"]["a_prop"] == 0.0 and fst["6"]["diff_prop"] == 0.25


def test_the_pool_scope_adds_every_dataset():
    pool = cmp.compare(ORIGINAL, HAIR_FREE)["scopes"]["pool"]
    assert pool["n"] == {"a": 8, "b": 8}
    assert pool["tone_distribution"]["bins"]["light"]["a_count"] == 5
    assert pool["flagged"]["a"] == {"n": 3, "n_artifact_detected": 1, "n_confidence_below_1": 3}
    assert pool["flagged"]["b"] == {"n": 1, "n_artifact_detected": 0, "n_confidence_below_1": 1}


def test_flagged_counts_are_reported_per_manifest_and_dataset():
    flagged = cmp.compare(ORIGINAL, HAIR_FREE)["scopes"]["ham"]["flagged"]
    assert flagged["a"] == {"n": 2, "n_artifact_detected": 1, "n_confidence_below_1": 2}
    assert flagged["b"] == {"n": 1, "n_artifact_detected": 0, "n_confidence_below_1": 1}


def test_a_manifest_without_the_flagged_block_reports_none():
    old = _manifest({"ham": _dataset({"dark": 1}, {"1": 1})})
    result = cmp.compare(old, HAM_ONLY)
    assert result["scopes"]["ham"]["flagged"]["a"] is None
    assert result["scopes"]["pool"]["flagged"]["a"] is None


def test_an_empty_distribution_has_no_proportions():
    empty = _manifest({"ham": _dataset({}, {})})
    bins = cmp.compare(empty, HAM_ONLY)["scopes"]["ham"]["tone_distribution"]["bins"]
    assert bins["dark"]["a_prop"] is None and bins["dark"]["diff_prop"] is None


def test_a_dataset_missing_from_one_manifest_is_rejected():
    with pytest.raises(ValueError, match="isic"):
        cmp.compare(ORIGINAL, HAM_ONLY)


def test_cli_writes_the_comparison_with_its_labels(tmp_path):
    paths = []
    for name, manifest in (("a.json", ORIGINAL), ("b.json", HAIR_FREE)):
        path = tmp_path / name
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "out" / "comparison.json"
    cmp.main([
        "--a", str(paths[0]), "--b", str(paths[1]), "--label-a", "original",
        "--label-b", "s2a", "--output-json", str(output),
    ])
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["labels"] == {"a": "original", "b": "s2a"}
    assert saved["sources"] == {"a": str(paths[0]), "b": str(paths[1])}
    assert saved["scopes"]["ham"]["tone_distribution"]["bins"]["dark"]["diff_count"] == -1
