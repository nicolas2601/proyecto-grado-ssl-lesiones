"""Aggregation of the three e3_final_eval.json (seeds 42, 43, 44) into e3_robustness.json."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e3_robustness_summary as script  # noqa: E402

from dermssl import e3_robustness as rob  # noqa: E402

SELECTION = "s" * 64


def contrast(metric, delta, low, high, subset="all"):
    return {"arm": "F4", "control": "B71", "metric": metric, "subset": subset, "point_delta": delta,
            "ci95_low": low, "ci95_high": high, "excludes_zero": not low <= 0 <= high,
            "direction": "arm_better" if delta > 0 else "control_better"}


def final_eval(seed, delta, low, high, *, selection=SELECTION, b71=0.71, f4=None):
    fingerprint = {"weights_sha256": f"w{seed}" * 8, "protocol": f"B_s{seed}_a1.0@224|cls4|balanced|tta"}
    if seed != 42:
        fingerprint["b71_seed"] = seed
    f4 = b71 + delta if f4 is None else f4
    return {"stage": "final", "selection_sha256": selection, "val_plan": ["B71", "F4"], "controls": ["B71"],
            "fingerprint": fingerprint,
            "rows": {"B71": {"macro_f1": b71, "balanced_accuracy": b71 - 0.02},
                     "F4": {"macro_f1": f4, "balanced_accuracy": f4 - 0.02}},
            "contrasts": [contrast("macro_f1", delta, low, high, "dark_tercile"),
                          contrast("macro_f1", delta, low, high),
                          contrast("balanced_accuracy", delta / 2, low / 2, high / 2)]}


def docs(*triples, **kw):
    return {seed: final_eval(seed, d, lo, hi, **kw) for seed, (d, lo, hi) in zip((42, 43, 44), triples, strict=True)}


WINS, NOISE = (0.036, 0.002, 0.078), (0.010, -0.020, 0.040)


@pytest.mark.parametrize("triples, reading", [
    ((WINS, WINS, WINS), "robust"),
    ((WINS, NOISE, WINS), "partial"),
    ((NOISE, NOISE, WINS), "partial"),
    ((NOISE, NOISE, NOISE), "none"),
])
def test_the_reading_follows_how_many_seeds_have_a_ci_that_excludes_zero(triples, reading):
    assert rob.summarize(docs(*triples), {s: "x" for s in (42, 43, 44)})["reading"] == reading


def test_reading_rule_counts_map_to_the_three_registered_strings():
    assert [rob.reading(n) for n in (3, 2, 1, 0)] == ["robust", "partial", "partial", "none"]


def test_each_seed_reports_the_rows_and_the_paired_contrast_as_computed():
    out = rob.summarize(docs(WINS, NOISE, WINS), {s: f"sha{s}" for s in (42, 43, 44)})
    s43 = out["seeds"]["43"]
    assert s43["B71"] == {"macro_f1": 0.71, "balanced_accuracy": pytest.approx(0.69)}
    assert s43["F4"]["macro_f1"] == pytest.approx(0.72)
    assert s43["macro_f1"] == {"delta": 0.010, "ci95": [-0.020, 0.040], "excludes_zero": False,
                               "direction": "arm_better"}
    assert s43["balanced_accuracy"]["delta"] == pytest.approx(0.005)
    assert out["seeds"]["42"]["macro_f1"]["excludes_zero"] is True
    assert out["seeds"]["44"]["weights_sha256"] == "w44" * 8
    assert out["inputs_sha256"] == {"42": "sha42", "43": "sha43", "44": "sha44"}


def test_mean_min_and_max_delta_over_the_three_seeds():
    out = rob.summarize(docs((0.03, 0.01, 0.05), (0.06, 0.01, 0.1), (0.00, -0.02, 0.02)), {s: "x" for s in (42, 43, 44)})
    assert out["delta_macro_f1"] == {"mean": pytest.approx(0.03), "min": 0.0, "max": 0.06}
    assert out["n_excluding_zero"] == 2 and out["reading"] == "partial"


def test_a_file_for_another_seed_than_its_slot_is_refused():
    bad = docs(WINS, WINS, WINS)
    bad[43]["fingerprint"]["b71_seed"] = 44
    with pytest.raises(rob.RobustnessInputError, match="43"):
        rob.summarize(bad, {s: "x" for s in (42, 43, 44)})


def test_the_three_files_must_come_from_the_same_pinned_selection():
    bad = docs(WINS, WINS, WINS)
    bad[44]["selection_sha256"] = "z" * 64
    with pytest.raises(rob.RobustnessInputError, match="selection"):
        rob.summarize(bad, {s: "x" for s in (42, 43, 44)})


def test_the_expected_selection_pin_is_enforced_when_given():
    with pytest.raises(rob.RobustnessInputError, match="pinned"):
        rob.summarize(docs(WINS, WINS, WINS), {s: "x" for s in (42, 43, 44)}, selection_sha256="y" * 64)


def test_the_same_weights_under_two_seeds_are_refused():
    bad = docs(WINS, WINS, WINS)
    bad[44]["fingerprint"]["weights_sha256"] = bad[43]["fingerprint"]["weights_sha256"]
    with pytest.raises(rob.RobustnessInputError, match="weights"):
        rob.summarize(bad, {s: "x" for s in (42, 43, 44)})


def test_a_missing_contrast_is_a_typed_error():
    bad = docs(WINS, WINS, WINS)
    bad[42]["contrasts"] = []
    with pytest.raises(rob.RobustnessInputError, match="macro_f1"):
        rob.summarize(bad, {s: "x" for s in (42, 43, 44)})


def write_inputs(tmp_path, triples):
    paths = {}
    for seed, doc in docs(*triples).items():
        paths[seed] = tmp_path / f"s{seed}.json"
        paths[seed].write_text(json.dumps(doc), encoding="utf-8")
    return paths


def run(tmp_path, paths, *extra):
    argv = [flag for seed, p in paths.items() for flag in (f"--s{seed}", str(p))]
    return script.main([*argv, "--out", str(tmp_path / "e3_robustness.json"), *extra])


def test_the_script_writes_the_summary_with_the_sha256_of_its_inputs(tmp_path):
    paths = write_inputs(tmp_path, (WINS, WINS, NOISE))
    assert run(tmp_path, paths, "--selection-sha256", SELECTION) == 0
    out = json.loads((tmp_path / "e3_robustness.json").read_text())
    assert out["reading"] == "partial"
    assert out["inputs_sha256"]["43"] == hashlib.sha256(paths[43].read_bytes()).hexdigest()


def test_the_script_aborts_with_code_2_on_a_bad_input_and_writes_nothing(tmp_path, capsys):
    paths = write_inputs(tmp_path, (WINS, WINS, WINS))
    assert run(tmp_path, paths, "--selection-sha256", "y" * 64) == 2
    assert "ABORT" in capsys.readouterr().out and not (tmp_path / "e3_robustness.json").exists()
