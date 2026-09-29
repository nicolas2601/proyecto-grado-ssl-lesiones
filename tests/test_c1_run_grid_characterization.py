"""Characterization of scripts/c1_run_grid.py: the refactored runner must behave
exactly like the pre-refactor one, byte for byte.

Why a differential test and not a checked-in golden file
--------------------------------------------------------
The saved proba come out of a sklearn logistic regression. Its last bits
depend on the BLAS build and on the CPU's SIMD path, so a digest recorded on
one machine is not a fact about the code on another (CI vs laptop vs pod). A
golden file would either be compared with a tolerance -- which is not
"bit-equal" -- or fail for reasons that have nothing to do with the refactor.

So the reference is the pre-refactor script itself, taken from git at
``REFERENCE_COMMIT`` (CI checks out with ``fetch-depth: 0``), loaded into a
temporary directory and run IN THE SAME PROCESS, on the same synthetic inputs
and the same output path, as the current one. Whatever the platform does to a
float, it does it to both sides.

What is pinned, per scenario (fresh run, full resume, partial resume, --limit)
------------------------------------------------------------------------------
* every file the grid writes, compared as raw bytes: each cell's
  ``.features.npy`` and ``.proba.npy`` (bit-equal arrays, dtype and shape
  included), each cell's manifest (all fields, including ``features_sha256``,
  ``train_features_sha256``, ``proba_sha256``, the gate blocks and metrics),
  ``c1_cells.json``, ``c1_bootstrap.json`` and both environment sidecars;
* stdout (the plan line, REUSADA/GATE/metric lines, the final table);
* the order in which backbones are built and cells are extracted, which is
  where PRIORITY_BACKBONES and the resume plan show up;
* every W&B call (config, metric keys and values, run names, group);
* the ``--help`` text and the abort messages of two refusal paths.

Volatile fields, listed explicitly
----------------------------------
* ``code_revision`` (in every manifest and W&B config) comes from ``git`` on
  the checkout, so both sides get the same pinned value through the module's
  ``code_revision`` attribute. Nothing else is volatile: the manifests carry
  no timestamps, and the sidecars' environment block (python, packages,
  thread pools, cpu count) is identical inside one process.

torch, the backbone builder and the per-image feature extractor are faked;
everything between them -- mask provenance, split/label alignment, geometry,
gating, probe, cell store, contrasts -- is the real code.

When a behaviour change is INTENDED, this test is expected to fail: update
``REFERENCE_COMMIT`` to the commit that introduced it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import c1_run_grid as current  # noqa: E402

import dermssl  # noqa: E402
from dermssl import c1_mask_provenance as mp  # noqa: E402
from dermssl.feature_cache import FeatureCache, fingerprint_ids  # noqa: E402
from dermssl.c1_mask_export import sha256_file  # noqa: E402

#: The last commit where scripts/c1_run_grid.py was the 500-line monolith.
REFERENCE_COMMIT = "bbc89a6"
SCRIPT = "scripts/c1_run_grid.py"
PINNED_REVISION = "0123456789abcdef0123456789abcdef01234567"

HEIGHT, WIDTH = 40, 48
N_CLASSES = 3
#: Mask coverage under which the fake extractor falls back to CLS.
SPECK = 0.01
TRAIN_IDS = [f"ISIC_{i:07d}" for i in range(36)]
VAL_IDS = [f"ISIC_{i:07d}" for i in range(100, 130)]
#: Deliberately NOT the priority order, so the plan has to reorder them.
BACKBONES = ["other_vitb14=arch_b:src_b", "ssl_proposed_vitb14=arch_a:src_a"]


# ---------------------------------------------------------------- reference

@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    try:
        source = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{REFERENCE_COMMIT}:{SCRIPT}"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        pytest.fail(f"no se pudo leer {SCRIPT} en {REFERENCE_COMMIT}: {error}")
    path = tmp_path_factory.mktemp("reference") / "scripts" / "c1_run_grid_reference.py"
    path.parent.mkdir()
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("c1_run_grid_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the world

def _write_world(root: Path) -> dict:
    rng = np.random.default_rng(7)
    from PIL import Image

    image_dir = root / "images"
    image_dir.mkdir()
    ids = TRAIN_IDS + VAL_IDS
    masks = []
    for n, image_id in enumerate(ids):
        pixels = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / f"{image_id}.jpg", quality=95)
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        y, x = 5 + n % 7, 4 + (3 * n) % 11
        if image_id == VAL_IDS[3]:
            # A speck: it survives every geometry, but the fake extractor
            # falls back to CLS on it, so mask_pool fails its gate and the
            # not_contrasted path is exercised next to real contrasts.
            mask[y:y + 2, x:x + 2] = 1
        else:
            mask[y:y + 12 + n % 9, x:x + 14 + n % 13] = 1
        masks.append(mask)

    # The artifact's order differs from the split's, so the position lookup matters.
    order = list(reversed(ids))
    masks_path = root / "masks.npy"
    np.save(masks_path, np.stack([masks[ids.index(i)] for i in order]))
    run, commit = mp.PUBLISHED_SEGMENTATION
    manifest = {
        "producer": mp.producer_for_run(run),
        "producer_is_label_free": True,
        "masks_sha256": sha256_file(masks_path),
        "checkpoint_sha256": mp.FINAL_STUDENT_R3_SHA256,
        "image_ids": order,
        "mask_shape": [HEIGHT, WIDTH],
        "segmentation_run": run,
        "segmentation_commit": commit,
        "inference": dict(mp.PUBLISHED_INFERENCE),
    }
    manifest_path = root / "masks_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    split_path = root / "split.json"
    split_path.write_text(json.dumps({"train_ids": TRAIN_IDS, "val_ids": VAL_IDS}), encoding="utf-8")

    cache_dir = root / "cache"
    cache = FeatureCache(cache_dir)
    first = BACKBONES[0].split("=")[0]
    for split, split_ids in (("train", TRAIN_IDS), ("val", VAL_IDS)):
        labels = np.arange(len(split_ids)) % N_CLASSES
        key = f"v1__{first}__{split}__r0__eval__{fingerprint_ids(split_ids)}"
        cache.save(key, np.zeros((len(split_ids), 2), dtype=np.float32), labels)

    return {
        "cache": cache_dir, "images": image_dir, "masks": masks_path,
        "manifest": manifest_path, "split": split_path, "out": root / "grid",
    }


def _argv(world: dict, *extra: str) -> list[str]:
    return [
        "c1_run_grid.py",
        "--cache-dir", str(world["cache"]), "--image-dir", str(world["images"]),
        "--masks", str(world["masks"]), "--masks-manifest", str(world["manifest"]),
        "--split-index", str(world["split"]), "--out-dir", str(world["out"]),
        "--backbones", *BACKBONES, *extra,
    ]


# ---------------------------------------------------------------- the fakes

class _Tensor:
    def __init__(self, image):
        self.image = image

    def to(self, device):
        return self


def _install_fakes(monkeypatch, log: list) -> None:
    ns = types.SimpleNamespace
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-fake"
    torch.version = ns(cuda=None)
    torch.backends = ns(cudnn=ns(is_available=lambda: False, version=lambda: None,
                                 deterministic=False, benchmark=False))
    torch.are_deterministic_algorithms_enabled = lambda: False
    torch.get_num_threads = lambda: 1
    torch.cuda = ns(is_available=lambda: False, is_initialized=lambda: False,
                    device_count=lambda: 0)
    torch.device = lambda name: f"device:{name}"
    # scipy/sklearn probe sys.modules["torch"] for array-API dispatch.
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)

    bank = types.ModuleType("cluster_frozen_bank")

    def parse_backbone_spec(raw):
        name, rest = raw.split("=", 1)
        arch, source = rest.split(":", 1)
        return name, arch, source

    def build_frozen_backbone(arch, source, resolution, device):
        log.append(("build", arch, source, resolution, device))
        model = ns(offset=float(sum(map(ord, arch + source)) % 17))
        # One backbone returns a dict, the other the bare model: both shapes exist.
        return {"model": model} if arch == "arch_a" else model

    bank.parse_backbone_spec = parse_backbone_spec
    bank.build_frozen_backbone = build_frozen_backbone
    monkeypatch.setitem(sys.modules, "cluster_frozen_bank", bank)

    tex = types.ModuleType("dermssl.torch_c1_extraction")
    tex.to_normalized_tensor = _Tensor

    def extract_feature(model, tensor, condition, mask):
        image = tensor.image.astype(np.float64)
        fallback = condition == "mask_pool" and (mask is None or mask.mean() < SPECK)
        region = image[mask] if (condition == "mask_pool" and not fallback) else image.reshape(-1, 3)
        coverage = 0.0 if mask is None else float(mask.mean())
        feature = np.concatenate([region.mean(axis=0) / 255.0, [coverage, model.offset / 17.0]])
        log.append(("extract", condition, round(float(feature.sum()), 6), bool(fallback)))
        return feature.astype(np.float32), fallback

    tex.extract_feature = extract_feature
    monkeypatch.setitem(sys.modules, "dermssl.torch_c1_extraction", tex)
    # `from dermssl import torch_c1_extraction` reads the package attribute
    # first, so a real module imported earlier in this worker would win.
    monkeypatch.setattr(dermssl, "torch_c1_extraction", tex, raising=False)

    wandb = types.ModuleType("wandb")

    class _Run:
        def __init__(self, kwargs):
            self.url = f"https://wandb.fake/{kwargs['name']}"

        def log(self, payload, step=None):
            log.append(("wandb.log", json.dumps(payload, sort_keys=True), step))

    def init(**kwargs):
        log.append(("wandb.init", json.dumps(kwargs, sort_keys=True, default=str)))
        return _Run(kwargs)

    wandb.init = init
    wandb.finish = lambda: log.append(("wandb.finish",))
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("NETRC", raising=False)


# ---------------------------------------------------------------- running

def _snapshot(out: Path) -> dict[str, bytes]:
    if not out.exists():
        return {}
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def _run(module, monkeypatch, capsys, argv, *, revision=PINNED_REVISION) -> dict:
    log: list = []
    _install_fakes(monkeypatch, log)
    monkeypatch.setattr(module, "code_revision", lambda: revision)
    monkeypatch.setattr(sys, "argv", list(argv))
    capsys.readouterr()
    try:
        status = module.main()
    except SystemExit as stop:
        status = f"exit:{stop.code}"
    captured = capsys.readouterr()
    return {"status": status, "stdout": captured.out, "log": log}


def _scenario(module, monkeypatch, capsys, world, *extra) -> list[dict]:
    """Fresh run, full resume, then a resume after one cell was invalidated."""
    shutil.rmtree(world["out"], ignore_errors=True)
    steps = []
    for step in ("fresh", "resume", "partial"):
        if step == "partial":
            next(world["out"].glob("cell__other_vitb14__bbox_crop.manifest.json")).unlink()
        result = _run(module, monkeypatch, capsys, _argv(world, *extra))
        result["files"] = _snapshot(world["out"])
        result["step"] = step
        steps.append(result)
    return steps


def _assert_same(reference_steps, current_steps):
    for ref, cur in zip(reference_steps, current_steps, strict=True):
        step = ref["step"]
        assert cur["status"] == ref["status"], step
        assert cur["stdout"] == ref["stdout"], step
        assert cur["log"] == ref["log"], step
        assert sorted(cur["files"]) == sorted(ref["files"]), step
        for name, payload in ref["files"].items():
            assert cur["files"][name] == payload, f"{step}: {name} cambio"


@pytest.fixture
def world(tmp_path):
    return _write_world(tmp_path)


# ---------------------------------------------------------------- the tests

def _assert_the_scenario_is_not_vacuous(steps) -> None:
    """Guards the guard: a scenario that silently did nothing would compare
    two empty directories and pass."""
    fresh, resume, partial = steps
    assert fresh["status"] == 0
    assert "celdas ya hechas: 0 | pendientes: 8" in fresh["stdout"]
    assert "celdas ya hechas: 8 | pendientes: 0" in resume["stdout"]
    assert "celdas ya hechas: 7 | pendientes: 1" in partial["stdout"]
    assert not [e for e in resume["log"] if e[0] == "extract"]
    builds = [e[1] for e in fresh["log"] if e[0] == "build"]
    assert builds[:4] == ["arch_a"] * 4, "el backbone prioritario no corrio primero"
    assert "FRENA" in fresh["stdout"] and "PASA" in fresh["stdout"]
    assert any(n.endswith(".proba.npy") for n in fresh["files"])
    assert "c1_bootstrap.environment.json" in fresh["files"]
    assert [e for e in fresh["log"] if e[0] == "wandb.init"]
    bootstrap = json.loads(fresh["files"]["c1_bootstrap.json"])
    assert bootstrap["results"] and bootstrap["not_contrasted"]
    manifest = json.loads(fresh["files"]["cell__ssl_proposed_vitb14__none.manifest.json"])
    assert manifest["code_revision"] == PINNED_REVISION


def test_a_full_grid_with_wandb_and_resume_is_unchanged(reference, monkeypatch, capsys, world):
    expected = _scenario(reference, monkeypatch, capsys, world, "--wandb-project", "p",
                         "--wandb-group", "g")
    _assert_the_scenario_is_not_vacuous(expected)
    actual = _scenario(current, monkeypatch, capsys, world, "--wandb-project", "p",
                       "--wandb-group", "g")
    _assert_same(expected, actual)


def test_a_limited_grid_with_another_probe_and_subset_of_conditions_is_unchanged(
        reference, monkeypatch, capsys, world):
    extra = ("--limit", "27", "--probe", "knn", "--conditions", "none", "bbox_crop", "mask_pool")
    expected = _scenario(reference, monkeypatch, capsys, world, *extra)
    actual = _scenario(current, monkeypatch, capsys, world, *extra)
    _assert_same(expected, actual)


def test_the_help_text_is_unchanged(reference, monkeypatch, capsys):
    expected = _run(reference, monkeypatch, capsys, ["c1_run_grid.py", "--help"])
    actual = _run(current, monkeypatch, capsys, ["c1_run_grid.py", "--help"])
    assert expected["status"] == "exit:0"
    assert actual == expected


@pytest.mark.parametrize("revision", ["unknown", PINNED_REVISION + "-dirty"])
def test_an_unusable_revision_aborts_the_same_way(reference, monkeypatch, capsys, world, revision):
    argv = _argv(world)
    expected = _run(reference, monkeypatch, capsys, argv, revision=revision)
    actual = _run(current, monkeypatch, capsys, argv, revision=revision)
    assert str(expected["status"]).startswith("exit:ABORT: code_revision")
    assert actual == expected
    assert not world["out"].exists()


def test_a_split_image_missing_from_the_artifact_aborts_the_same_way(
        reference, monkeypatch, capsys, world):
    manifest = json.loads(world["manifest"].read_text(encoding="utf-8"))
    split = {"train_ids": TRAIN_IDS + ["ISIC_9999999"], "val_ids": VAL_IDS}
    world["split"].write_text(json.dumps(split), encoding="utf-8")
    cache = FeatureCache(world["cache"])
    first = BACKBONES[0].split("=")[0]
    for stale in world["cache"].glob(f"v1__{first}__train__*"):
        stale.unlink()
    cache.save(f"v1__{first}__train__r0__eval__{fingerprint_ids(split['train_ids'])}",
               np.zeros((37, 2), dtype=np.float32), np.arange(37) % N_CLASSES)
    assert "ISIC_9999999" not in manifest["image_ids"]

    expected = _run(reference, monkeypatch, capsys, _argv(world))
    actual = _run(current, monkeypatch, capsys, _argv(world))
    assert "no estan en el artefacto" in str(expected["status"])
    assert actual == expected
