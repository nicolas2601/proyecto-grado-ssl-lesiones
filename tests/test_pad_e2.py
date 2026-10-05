"""E2 (docs/33): frozen HAM probe on PAD target_test BCC/MEL/NV; D3 - D2 paired and patient-grouped."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dermssl import pad_e2 as e2
from dermssl import pad_target_split as ts
from dermssl import pad_ufes_readings as pr

N_RES = 40


def _labels(n_patients: int = 60, per_patient: int = 4):
    classes = ["bcc", "mel", "nv", "akiec"]
    labels = [classes[(p + i) % 4] for p in range(n_patients) for i in range(per_patient)]
    groups = [f"PAT_{p}" for p in range(n_patients) for _ in range(per_patient)]
    return np.array(labels), np.array(groups)


def _probas(labels, signal: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    proba = rng.dirichlet(np.ones(7), size=len(labels))
    for row, label in zip(proba, labels):
        row[pr.HAM_CLASSES.index(label)] += signal
    return proba / proba.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def evaluated():
    labels, groups = _labels()
    probas = {"D0": _probas(labels, 0.2, 1), "D1": _probas(labels, 0.4, 2),
              "D2": _probas(labels, 0.4, 3), "D3": _probas(labels, 3.0, 4)}
    return e2.evaluate(labels, groups, probas, n_resamples=N_RES), labels


class TestEvaluate:
    def test_only_the_primary_reading_enters(self, evaluated):
        doc, labels = evaluated
        assert doc["classes"] == ["bcc", "mel", "nv"]
        assert doc["n_images"] == int(np.isin(labels, ["bcc", "mel", "nv"]).sum())

    def test_every_arm_reports_the_requested_metrics_with_intervals(self, evaluated):
        doc, _ = evaluated
        assert list(doc["arms"]) == ["D0", "D1", "D2", "D3"]
        for arm in doc["arms"].values():
            assert {"macro_f1", "balanced_accuracy", "auc", "mcc"} <= arm.keys()
            assert set(arm["ci95"]) >= {"macro_f1", "balanced_accuracy", "auc", "mcc"}

    def test_the_resampling_unit_is_the_patient(self, evaluated):
        doc, _ = evaluated
        assert doc["arms"]["D3"]["bootstrap"]["unit"] == "patient_id"
        assert doc["bootstrap"]["unit"] == "patient_id" and doc["bootstrap"]["rng_seed"] == 20260923

    def test_the_primary_contrast_is_d3_minus_d2_on_macro_f1(self, evaluated):
        doc, _ = evaluated
        primary = doc["primary_contrast"]
        assert (primary["arm"], primary["control"], primary["metric"]) == ("D3", "D2", "macro_f1")
        assert primary["direction"] == "arm_better" and doc["prediction_d3_minus_d2_met"] is True

    def test_all_pairs_are_contrasted_on_every_metric(self, evaluated):
        doc, _ = evaluated
        pairs = {(r["arm"], r["control"]) for r in doc["paired"]}
        assert pairs == set(e2.CONTRASTS) and len(doc["paired"]) == len(e2.CONTRASTS) * 5

    def test_the_outside_rate_is_descriptive_per_arm(self, evaluated):
        doc, _ = evaluated
        assert set(doc["outside_rate"]) == set(e2.ARMS)

    def test_an_interval_that_includes_zero_does_not_meet_the_prediction(self):
        labels, groups = _labels()
        same = _probas(labels, 1.0, 7)
        doc = e2.evaluate(labels, groups, {a: same for a in e2.ARMS}, n_resamples=N_RES)
        assert doc["primary_contrast"]["direction"] == "indistinguishable"
        assert doc["prediction_d3_minus_d2_met"] is False

    def test_the_default_bootstrap_is_the_projects(self):
        assert e2.N_RESAMPLES == pr.BOOTSTRAP["n_resamples"] and e2.RNG_SEED == pr.BOOTSTRAP["rng_seed"]


class TestNonFinite:
    def test_a_resample_with_one_class_aborts_instead_of_dropping_it_silently(self):
        labels = np.array(["bcc", "bcc", "mel", "nv"])
        groups = np.array(["P0", "P0", "P1", "P1"])
        probas = {arm: _probas(labels, 1.0, 5) for arm in e2.ARMS}
        with pytest.raises(e2.NonFiniteResamplesError, match="auc"):
            e2.evaluate(labels, groups, probas, n_resamples=N_RES)


class TestCanonicalCompute:
    def test_cuda_with_batch_64_passes(self):
        e2.check_canonical_compute("cuda", 64)

    @pytest.mark.parametrize("device,batch", [("cpu", 64), ("cuda", 16)])
    def test_anything_else_is_refused(self, device, batch):
        with pytest.raises(e2.NonCanonicalComputeError):
            e2.check_canonical_compute(device, batch)


def _run(**over):
    base = {"code_revision": "a" * 40, "exclude_ids_sha256": "e" * 64, "seed": 42, "split_hash": "0" * 64,
            "method": "simsiam_dinov2", "backbone": "dinov2_vitb14", "initial_weights": "w.pt",
            "max_steps": 2000, "precision": "bf16", "hyperparameters": {"effective_lr": 1.0},
            "adaptation_levers": {"anchor_l2sp": 0.0001}, "checkpoint_fractions": [0.25, 1.0]}
    return {**base, **over}


class TestRunConsistency:
    def test_matching_runs_pass_and_public_has_none(self):
        e2.check_d_runs({"D1": _run(code_revision="b" * 40), "D2": _run(), "D3": _run()})

    @pytest.mark.parametrize("field,value", [("code_revision", "c" * 40), ("max_steps", 2500), ("seed", 43),
                                              ("hyperparameters", {"effective_lr": 2.0}),
                                              ("initial_weights", "other.pt")])
    def test_d2_and_d3_must_agree(self, field, value):
        with pytest.raises(e2.RunMismatchError, match=field):
            e2.check_d_runs({"D1": _run(), "D2": _run(), "D3": _run(**{field: value})})

    def test_the_exclusion_list_must_be_the_d1_clean_one(self):
        with pytest.raises(e2.RunMismatchError, match="exclude_ids_sha256"):
            e2.check_d_runs({"D1": _run(), "D2": _run(exclude_ids_sha256="f" * 64), "D3": _run(exclude_ids_sha256="f" * 64)})

    def test_a_missing_run_is_refused(self):
        with pytest.raises(e2.RunMismatchError, match="D3"):
            e2.check_d_runs({"D1": _run(), "D2": _run()})


def _manifest_rows(tmp_path, n_patients: int = 12):
    rows = []
    for p in range(n_patients):
        for i in range(2):
            name = f"PAT_{p}_{i}_{p}{i}"
            path = tmp_path / "images" / f"{name}.png"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
            rows.append({"img_id": name, "patient_id": f"PAT_{p}", "lesion_id": str(i),
                         "diagnostic": "BCC", "ham_label": ["bcc", "mel", "nv"][(p + i) % 3],
                         "path": f"images/{name}.png", "sha256": hashlib.sha256(name.encode()).hexdigest()})
    return rows


def _split(tmp_path, rows):
    doc = ts.build_document(rows, lambda r: r["sha256"], manifest_sha256="m" * 64)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(doc))
    return path


class _Spy(dict):
    """A manifest row that records whether the label was read before the predictions existed."""

    def __init__(self, row, log):
        super().__init__(row)
        self.log = log

    def __getitem__(self, key):
        if key == "ham_label":
            self.log.append("label")
        return super().__getitem__(key)


class TestSingleEvaluation:
    def _predict(self, log):
        def predict(ids, paths):
            log.append("predict")
            assert set(paths) == set(ids)
            rng = np.random.RandomState(0)
            return {arm: rng.dirichlet(np.ones(7), size=len(ids)) for arm in e2.ARMS}
        return predict

    def test_it_locks_target_test_freezes_probas_then_reads_labels(self, tmp_path):
        log, rows = [], _manifest_rows(tmp_path)
        split = _split(tmp_path, rows)
        spied = [_Spy(r, log) for r in rows]
        doc = e2.run_single_evaluation(split, spied, tmp_path, self._predict(log), tmp_path / "out.json",
                                       n_resamples=N_RES)
        assert log.index("predict") < log.index("label")
        assert (tmp_path / ts.LOCK_NAME).is_file()
        frozen = tmp_path / "out.probas.npz"
        assert doc["probas_sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
        assert doc["target_test_sha256"] == json.loads(split.read_text())["sha256"]["target_test"]

    def test_only_target_test_images_are_predicted(self, tmp_path):
        log, rows = [], _manifest_rows(tmp_path)
        split = _split(tmp_path, rows)
        seen = {}
        def predict(ids, paths):
            seen["ids"] = list(ids)
            return {arm: np.full((len(ids), 7), 1 / 7) for arm in e2.ARMS}
        e2.run_single_evaluation(split, rows, tmp_path, predict, tmp_path / "out.json", n_resamples=N_RES)
        assert sorted(seen["ids"]) == sorted(json.loads(split.read_text())["target_test"]["img_ids"])

    def test_a_second_evaluation_refuses(self, tmp_path):
        rows = _manifest_rows(tmp_path)
        split = _split(tmp_path, rows)
        e2.run_single_evaluation(split, rows, tmp_path, self._predict([]), tmp_path / "a.json", n_resamples=N_RES)
        with pytest.raises(ts.TargetTestAlreadyReadError):
            e2.run_single_evaluation(split, rows, tmp_path, self._predict([]), tmp_path / "b.json", n_resamples=N_RES)

    def test_a_changed_image_aborts_before_any_prediction(self, tmp_path):
        log, rows = [], _manifest_rows(tmp_path)
        split = _split(tmp_path, rows)
        victim = json.loads(split.read_text())["target_test"]["img_ids"][0]
        (tmp_path / "images" / f"{victim}.png").write_bytes(b"changed")
        with pytest.raises(e2.ImageChangedError, match=victim):
            e2.run_single_evaluation(split, rows, tmp_path, self._predict(log), tmp_path / "out.json", n_resamples=N_RES)
        assert "predict" not in log
        assert not (tmp_path / ts.LOCK_NAME).exists()

    def test_predictions_in_another_order_or_shape_are_refused(self, tmp_path):
        rows = _manifest_rows(tmp_path)
        split = _split(tmp_path, rows)
        with pytest.raises(e2.E2Error, match="probabilidades"):
            e2.run_single_evaluation(split, rows, tmp_path, lambda ids, paths: {"D0": np.zeros((1, 7))},
                                     tmp_path / "out.json", n_resamples=N_RES)
