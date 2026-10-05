"""scripts/pad_e2_eval.py refuses before touching any model, image or label."""

from __future__ import annotations

import pytest

from dermssl import pad_target_split as ts
from dermssl import run_provenance as rp
from scripts import pad_e2_eval as script

SPECS = ["--d1", "dinov2_vitb14:a.pt", "--d2", "dinov2_vitb14:b.pt", "--d3", "dinov2_vitb14:c.pt"]


def _argv(tmp_path, *extra):
    return ["--split", str(tmp_path / "split.json"), "--manifest", str(tmp_path / "m.json"),
            "--pad-root", str(tmp_path), "--ham-root", str(tmp_path / "ham"), "--out", str(tmp_path / "out.json"),
            *SPECS, *extra]


def test_it_requires_the_single_evaluation_flag(tmp_path):
    with pytest.raises(SystemExit, match="single-evaluation"):
        script.main(_argv(tmp_path))


def test_it_refuses_an_existing_output(tmp_path):
    (tmp_path / "out.json").write_text("{}")
    with pytest.raises(SystemExit, match="ya existe"):
        script.main(_argv(tmp_path, "--single-evaluation"))


def test_it_refuses_when_target_test_was_already_read(tmp_path):
    (tmp_path / ts.LOCK_NAME).write_text("{}")
    with pytest.raises(SystemExit, match="target_test ya se leyo"):
        script.main(_argv(tmp_path, "--single-evaluation"))


def test_d0_must_be_the_public_model(tmp_path):
    with pytest.raises(SystemExit, match="D0"):
        script.main(_argv(tmp_path, "--single-evaluation", "--d0", "dinov2_vitb14:x.pt"))


def test_a_dirty_tree_refuses(tmp_path, monkeypatch):
    def dirty(repo):
        raise rp.DirtyTreeError("arbol sucio")
    monkeypatch.setattr(rp, "clean_revision", dirty)
    with pytest.raises(SystemExit, match="arbol sucio"):
        script.main(_argv(tmp_path, "--single-evaluation"))


def test_d1_is_required_in_both_modes(tmp_path):
    argv = _argv(tmp_path)
    del argv[argv.index("--d1"):argv.index("--d1") + 2]
    with pytest.raises(SystemExit):
        script.parse_args(argv)


def test_the_lock_is_created_next_to_the_split_and_never_by_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "clean_revision", lambda repo: "a" * 40)
    assert script.preflight(script.parse_args(_argv(tmp_path, "--single-evaluation"))) == "a" * 40
    assert not (tmp_path / ts.LOCK_NAME).exists()


class TestWiring:
    """``main`` end to end with the model-facing calls replaced: only the wiring is under test."""

    @pytest.fixture
    def wired(self, tmp_path, monkeypatch):
        import hashlib
        import json

        import numpy as np

        from dermssl import pad_ufes_readings as pr

        rows = []
        for p in range(12):
            for i in range(2):
                name = f"PAT_{p}_{i}_{p}{i}"
                (tmp_path / f"{name}.png").write_bytes(name.encode())
                rows.append({"img_id": name, "patient_id": f"PAT_{p}", "lesion_id": str(i), "diagnostic": "BCC",
                             "ham_label": ["bcc", "mel", "nv"][(p + i) % 3], "path": f"{name}.png",
                             "sha256": hashlib.sha256(name.encode()).hexdigest()})
        (tmp_path / "m.json").write_text(json.dumps({"images": rows}))
        doc = ts.build_document(rows, lambda r: r["sha256"], manifest_sha256="m" * 64)
        (tmp_path / "split.json").write_text(json.dumps(doc))
        for name in "abc":
            (tmp_path / f"{name}.pt").write_bytes(name.encode())
        run = {"code_revision": "a" * 40, "exclude_ids_sha256": "e" * 64, "seed": 42, "max_steps": 2000}
        monkeypatch.setattr(rp, "clean_revision", lambda repo: "a" * 40)
        monkeypatch.setattr(script.c1, "run_manifests", lambda specs: {a: dict(run) for a in ("D1", "D2", "D3")})
        split = {"train_ids": ["h1", "h2", "h3"], "val_ids": ["v1"], "classes": list(pr.HAM_CLASSES),
                 "id2label": {"h1": 0, "h2": 1, "h3": 2}}
        monkeypatch.setattr(script, "rebuild_split", lambda root: (split, None))
        monkeypatch.setattr(script.fe, "check_val_protocol", lambda train, val: None)
        monkeypatch.setattr(script.bank, "index_images", lambda roots: {})
        monkeypatch.setattr(script.bank, "build_frozen_backbone", lambda *a: {"model": object()})
        monkeypatch.setattr(script.bank, "extract_features",
                            lambda model, ids, *a: (np.zeros((len(ids), 4)), np.zeros(len(ids))))
        rng = np.random.RandomState(0)
        monkeypatch.setattr(script.probes, "run_probe",
                            lambda probe, x_fit, y_fit, x_test, n: rng.dirichlet(np.ones(n), size=len(x_test)))
        argv = ["--single-evaluation", "--split", str(tmp_path / "split.json"), "--manifest", str(tmp_path / "m.json"),
                "--pad-root", str(tmp_path), "--ham-root", str(tmp_path / "ham"),
                "--out", str(tmp_path / "res" / "pad_e2_eval.json"), "--device", "cuda",
                "--d1", f"dinov2_vitb14:{tmp_path / 'a.pt'}", "--d2", f"dinov2_vitb14:{tmp_path / 'b.pt'}",
                "--d3", f"dinov2_vitb14:{tmp_path / 'c.pt'}"]
        return argv, tmp_path

    def test_it_writes_the_result_the_sidecar_the_frozen_probas_and_the_lock(self, wired):
        import json

        argv, tmp = wired
        assert script.main(argv) == 0
        out = tmp / "res" / "pad_e2_eval.json"
        doc = json.loads(out.read_text())
        assert doc["primary_contrast"]["arm"] == "D3" and doc["compute"]["batch_size"] == 64
        assert list(doc) == [  # the real-mode document, key for key and in order (regression of the rehearsal change)
            "code_revision", "protocol", "probe", "specs", "backbone_sha256", "runs", "split_sha256",
            "split_manifest_sha256", "manifest_sha256", "n_fit", "compute", "dataset_doi", "license",
            "target_test_sha256", "n_target_test_images", "probas_sha256", "reading", "classes", "n_images",
            "n_patients", "bootstrap", "arms", "paired", "primary_contrast", "prediction_d3_minus_d2_met",
            "outside_rate"]
        assert set(doc["split_sha256"]) == {"target_adapt", "target_test"}
        assert set(doc["backbone_sha256"]) == {"D1", "D2", "D3"}
        assert out.with_suffix(".probas.npz").is_file() and (tmp / ts.LOCK_NAME).is_file()
        assert (tmp / "res" / "pad_e2_eval.environment.json").is_file()

    def test_the_second_run_refuses(self, wired):
        argv, _ = wired
        script.main(argv)
        argv[argv.index("--out") + 1] += ".again.json"
        with pytest.raises(SystemExit, match="target_test ya se leyo"):
            script.main(argv)

    def test_rehearsal_end_to_end_uses_only_d0_d1_and_leaves_target_test_alone(self, wired, monkeypatch):
        import json

        argv, tmp = wired
        built = []
        monkeypatch.setattr(script.bank, "build_frozen_backbone", lambda *a: built.append(a) or {"model": object()})
        argv = [a for a in argv if a != "--single-evaluation"]
        argv = argv[:argv.index("--d2")] + ["--rehearsal"]
        argv[argv.index("--out") + 1] = str(tmp / "pad_rehearsal" / "e2_rehearsal.json")
        assert script.main(argv) == 0
        out = tmp / "pad_rehearsal" / "e2_rehearsal.json"
        doc = json.loads(out.read_text())
        assert doc["rehearsal"] is True and doc["split"] == "target_adapt" and doc["discard"] is True
        assert list(doc["arms"]) == ["D0", "D1"] and len(built) == 2
        assert (tmp / "pad_rehearsal" / "e2_rehearsal.environment.json").is_file()
        assert not (tmp / ts.LOCK_NAME).exists()


class TestRehearsalMode:
    def _argv(self, tmp_path, *extra, out="pad_rehearsal/e2_rehearsal.json"):
        return ["--split", str(tmp_path / "split.json"), "--manifest", str(tmp_path / "m.json"),
                "--pad-root", str(tmp_path), "--ham-root", str(tmp_path / "ham"), "--out", str(tmp_path / out),
                "--d1", "dinov2_vitb14:a.pt", *extra]

    def test_the_two_modes_exclude_each_other(self, tmp_path):
        with pytest.raises(SystemExit):
            script.parse_args(self._argv(tmp_path, "--rehearsal", "--single-evaluation"))

    def test_one_of_the_two_modes_is_required(self, tmp_path):
        with pytest.raises(SystemExit, match="single-evaluation o --rehearsal"):
            script.main(self._argv(tmp_path))

    def test_the_real_mode_still_requires_d2_and_d3(self, tmp_path):
        with pytest.raises(SystemExit, match="--d2 y --d3"):
            script.main(self._argv(tmp_path, "--single-evaluation"))

    def test_rehearsal_needs_only_d1_and_never_consults_the_lock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rp, "clean_revision", lambda repo: "a" * 40)
        lock = tmp_path / ts.LOCK_NAME
        lock.write_text("untouched")
        args = script.parse_args(self._argv(tmp_path, "--rehearsal"))
        assert script.preflight(args) == "a" * 40
        assert lock.read_text() == "untouched"

    @pytest.mark.parametrize("out", ["pad_da/e2_rehearsal.json", "pad_rehearsal/pad_e2_eval.json",
                                     "resultados/pad_rehearsal/e2_rehearsal.json"])
    def test_rehearsal_refuses_an_output_outside_a_rehearsal_directory(self, tmp_path, out):
        with pytest.raises(SystemExit, match="rehearsal"):
            script.main(self._argv(tmp_path, "--rehearsal", out=out))

    def test_rehearsal_still_refuses_a_dirty_tree(self, tmp_path, monkeypatch):
        def dirty(repo):
            raise rp.DirtyTreeError("arbol sucio")
        monkeypatch.setattr(rp, "clean_revision", dirty)
        with pytest.raises(SystemExit, match="arbol sucio"):
            script.main(self._argv(tmp_path, "--rehearsal"))
