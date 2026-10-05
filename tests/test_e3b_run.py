"""scripts/e3b_run.py end to end with fake backbones over a real (fake-extractor) E3 run."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e3b_run  # noqa: E402
import e3b_support as sup  # noqa: E402
import test_e3b_extraction as tx  # noqa: E402

from dermssl import e3_store as st  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl.e3b import backbones as bb  # noqa: E402
from dermssl.e3b.runtime import Runtime  # noqa: E402

CLASSES = ("a", "b", "c")


@pytest.fixture(scope="module")
def e3(tmp_path_factory):
    out = tmp_path_factory.mktemp("e3")
    path, sha = sup.build_e3_dir(out)
    return out, path, sha


class Builds:
    def __init__(self, fail=()):
        self.names, self.fail, self.failed_calls = [], set(fail), 0

    def __call__(self, name):
        self.names.append(name)

        def extract(images):
            if name in self.fail:
                self.failed_calls += 1
                raise RuntimeError("cuda out of memory")
            flat = images.reshape(len(images), -1, 3).astype(np.float32)
            scale = 1 + len(name) / 10
            return (np.concatenate([flat.mean(axis=1), flat.std(axis=1)], axis=1) * scale).astype(np.float32)

        return bb.Backbone(name, 6, None, tx.resize_crop, extract, hashlib.sha256(name.encode()).hexdigest(),
                           "bilinear")


def runtime(builds=None, labels_tripwire=False):
    ids = {s: [f"{s}_{i}" for i in range(sup.SIZES[s])] for s in ("train", "val")}

    def labels(split):
        assert not labels_tripwire, "labels read before the gate"
        return np.array([sup.label_of(i) for i in ids[split]])

    rt = Runtime(ids=ids, labels_of=labels, groups_of=lambda s: np.array([f"{s}{k // 2}" for k in range(len(ids[s]))]),
                 classes=CLASSES, mel=1, load_item=sup.t3.load_item, hair_fn=None,
                 build_backbone=builds or Builds(), fingerprint={"masks_sha256": "m" * 64, "holdout_sha256": "h" * 64})
    return rt


def cli(out, e3, *extra, rt=None):
    e3_dir, selection, sha = e3
    argv = ["--out-dir", str(out), "--e3-dir", str(e3_dir), "--selection", str(selection),
            "--selection-sha256", sha, "--ci-resamples", "12", "--batch-size", "5", *extra]
    return e3b_run.main(argv, runtime=rt or runtime(), revision="rev0")


def summary(out):
    return json.loads((out / "e3b_summary.json").read_text())


def test_full_run_writes_rows_contrasts_probas_and_sidecars(e3, tmp_path, capsys):
    assert cli(tmp_path, e3, "--models", "alexnet", "resnet50") == 0
    doc = summary(tmp_path)
    assert doc["complete"] and doc["winner"] == "F4" and doc["selection_sha256"] == e3[2] and not doc["smoke"]
    assert set(doc["models"]) == {"alexnet", "resnet50"} and doc["n"] == {"train": 36, "val": 18}
    assert set(doc["models"]["alexnet"]["rows"]) == {"full", "F4"}
    assert set(doc["proposed"]["rows"]) == {"B71", "B71+F4", "B71_tta", "B71+F4_tta"}
    assert doc["proposed"]["check_against_e3_final"]["matches"]
    assert len(doc["contrasts"]) == 2 * 5 * 3 and {c["family"] for c in doc["contrasts"]} >= {"seg_effect"}
    assert doc["protocol"]["bootstrap"]["grouped_by"] == "lesion_id" and doc["failures"] == []
    with np.load(tmp_path / "e3b_probas.npz") as data:
        assert {"y_val", "val_ids", "alexnet", "alexnet+F4", "B71", "B71+F4_tta"} <= set(data.files)
        assert data["alexnet"].shape == (18, 3)
    manifest = json.loads((tmp_path / "e3b_probas.json").read_text())
    assert manifest["sha256"] == st.file_sha256(tmp_path / "e3b_probas.npz") and manifest["arrays"]["alexnet"] == [18, 3]
    for artifact in ("e3b_summary.json", "e3b_probas.json", "cells/e3b__eval__alexnet.json",
                     "cells/e3b__extract__resnet50__val.json", "cells/e3b__b71.json", "cells/e3b__contrasts.json"):
        path = tmp_path / artifact
        assert run_environment.read_sidecar(path) is not None, artifact
        assert run_environment.sidecar_problems(run_environment.sidecar_path(path)) == []


def test_the_two_arms_share_the_validation_set_and_the_f4_arm_differs(e3, tmp_path):
    assert cli(tmp_path, e3, "--models", "alexnet") == 0
    with np.load(tmp_path / "e3b_probas.npz") as data:
        assert not np.allclose(data["alexnet"], data["alexnet+F4"])
        assert list(data["val_ids"]) == [f"val_{i}" for i in range(18)]


def test_a_second_run_skips_every_cell_and_never_loads_a_backbone(e3, tmp_path, capsys):
    assert cli(tmp_path, e3, "--models", "alexnet") == 0
    first = (tmp_path / "e3b_summary.json").read_bytes()
    builds = Builds()
    capsys.readouterr()
    assert cli(tmp_path, e3, "--models", "alexnet", rt=runtime(builds)) == 0
    out = capsys.readouterr().out
    assert "done" not in out and out.count("skipped") >= 7 and builds.names == []
    assert (tmp_path / "e3b_summary.json").read_bytes() == first


def test_a_failing_backbone_is_retried_listed_and_the_others_finish(e3, tmp_path, capsys):
    builds = Builds(fail={"resnet50"})
    assert cli(tmp_path, e3, "--models", "alexnet", "resnet50", rt=runtime(builds)) == 1
    assert "[FALLO] e3b__extract__resnet50__train" in capsys.readouterr().out
    doc = summary(tmp_path)
    assert not doc["complete"] and set(doc["models"]) == {"alexnet"}
    assert [f["cell"] for f in doc["failures"]] == [f"e3b__extract__resnet50__{s}" for s in ("train", "val")]
    assert builds.failed_calls == 2 * 3 and builds.names.count("resnet50") == 1  # 2 splits x (1 try + 2 retries)


def test_a_deleted_probas_file_is_recomputed_on_resume(e3, tmp_path):
    assert cli(tmp_path, e3, "--models", "alexnet") == 0
    (tmp_path / "probas" / "alexnet.npz").unlink()
    assert cli(tmp_path, e3, "--models", "alexnet") == 0
    assert (tmp_path / "probas" / "alexnet.npz").is_file()


def test_the_gate_refuses_before_anything_is_loaded_or_read(e3, tmp_path):
    rt = runtime(labels_tripwire=True)
    e3_dir, selection, _ = e3
    argv = ["--out-dir", str(tmp_path), "--e3-dir", str(e3_dir), "--selection", str(selection),
            "--selection-sha256", "0" * 64]
    assert e3b_run.main(argv, runtime=rt, revision="rev0") == 2
    assert not (tmp_path / "cells").exists()
    argv[-1] = e3[2]
    assert e3b_run.main([*argv, "--expected-winner", "F1"], runtime=rt, revision="rev0") == 2


def test_the_default_pin_is_the_committed_one(e3, tmp_path):
    args = e3b_run.parse_args(["--out-dir", str(tmp_path)])
    assert args.selection_sha256 == "2446c05a18e6e76d16d9953d89a8d28547cde61a6c25cc64aa60fb92c0563777"
    assert args.expected_winner == "F4" and args.selection.name == "e3_selection.json"
    assert args.models == list(e3b_run.proto.MODELS)


def test_without_e3_features_only_a_smoke_may_run(e3, tmp_path):
    e3_dir, selection, sha = e3
    argv = ["--out-dir", str(tmp_path), "--selection", str(selection), "--selection-sha256", sha,
            "--models", "alexnet", "--ci-resamples", "5"]
    assert e3b_run.main(argv, runtime=runtime(), revision="rev0") == 2
    assert e3b_run.main([*argv, "--limit", "70"], runtime=runtime(), revision="rev0") == 0
    doc = summary(tmp_path)
    assert doc["smoke"] and doc["proposed"] is None and doc["contrasts"] == [] and doc["complete"]


def test_a_b71_that_does_not_reproduce_e3_stops_the_run_before_any_backbone(e3, tmp_path):
    e3_dir, selection, sha = e3
    copy = tmp_path / "e3copy"
    shutil.copytree(e3_dir, copy)
    final = json.loads((copy / "e3_final_eval.json").read_text())
    final["rows"]["F4"]["macro_f1"] += 0.1
    st.write_json_atomic(copy / "e3_final_eval.json", final)
    builds = Builds()
    code = cli(tmp_path / "out", (copy, selection, sha), "--models", "alexnet", rt=runtime(builds))
    assert code == 1 and builds.names == [] and not summary(tmp_path / "out")["complete"]


def test_e3_features_of_other_ids_are_refused(e3, tmp_path):
    rt = runtime()
    rt.ids["val"] = rt.ids["val"][:-1]
    assert cli(tmp_path, e3, "--models", "alexnet", rt=rt) == 1
    assert not summary(tmp_path)["complete"]
