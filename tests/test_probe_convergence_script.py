"""probe_convergence_check.py end to end on a fake published run (kept features, no GPU)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dermssl import t4_sog as t4
from dermssl import v71_grid as vg
from dermssl import v71_metrics as vm
from dermssl import v71_protocol as vp

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("cls", "cls4")
N_CLASSES = 3


def _load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("probe_convergence_check", ROOT / "scripts/probe_convergence_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parts(rng, n, n_views, y=None):
    parts = {p: rng.normal(size=(n_views, n, 3)).astype(np.float32) for p in ("cls", "cls4")}
    if y is not None:
        parts["cls"][0] += y[:, None]
    return parts


def _published(tmp_path: Path, y_fit, y_hold):
    rng = np.random.default_rng(0)
    units, rows = tmp_path / "salidas/v71/units", {}
    for name in t4.WEIGHTS.values():
        udir = units / f"{name}@224"
        udir.mkdir(parents=True)
        fit, hold = _parts(rng, len(y_fit), 1, y_fit), _parts(rng, len(y_hold), 2)
        np.savez(udir / "features.npz", **{f"fit_{k}": v for k, v in fit.items()},
                 **{f"hold_{k}": v for k, v in hold.items()})
        probas = vg.unit_probas(fit, y_fit, hold, N_CLASSES, features=FEATURES)
        cells = [vp.config_id(name, 224, *key) for key in probas]
        np.savez(udir / "probas.npz", **{f"p{k}": probas[key].astype(np.float32) for k, key in enumerate(probas)})
        (udir / "result.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
        rows.update({cid: vm.metric_bundle(y_hold, probas[key], mel=0) for cid, key in zip(cells, probas)})
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return units, grid


@pytest.fixture
def run(tmp_path, monkeypatch):
    script = _load_script()
    y_fit, y_hold = np.arange(60) % N_CLASSES, np.arange(24) % N_CLASSES
    ids = {"fit": [f"f{i}" for i in range(60)], "holdout": [f"h{i}" for i in range(24)]}
    labels = {**dict(zip(ids["fit"], y_fit)), **dict(zip(ids["holdout"], y_hold))}
    ctx = {"ids": ids, "id2label": labels, "classes": ["mel", "nv", "bcc"], "mel": 0, "holdout_sha": "x"}
    monkeypatch.setattr(script.io, "load_context", lambda args: ctx)
    monkeypatch.setattr(script.io, "WeightBank", lambda path: None)
    monkeypatch.setattr(script.pc.vp, "FEATURES", FEATURES)
    units, grid = _published(tmp_path, y_fit, y_hold)
    out = tmp_path / "diag"
    argv = [f"--{k}={tmp_path / k}" for k in ("ham-root", "image-dir", "split-index", "holdout", "weights-dir",
                                               "inputs-dir")]
    argv += [f"--reference-units={units}", f"--features-dir={units}", f"--v71-grid={grid}", f"--out-dir={out}",
             "--jobs=1", "--allow-dirty"]
    return script, argv, units, out


def test_the_refit_of_the_published_run_passes_and_writes_json_readme_and_sidecar(run):
    script, argv, _, out = run
    assert script.main(argv) == 0
    doc = json.loads((out / "probe_convergence.json").read_text(encoding="utf-8"))
    assert doc["sanity"]["passed"] and doc["summary"]["n_fits"] == 3 * len(FEATURES) * 2
    assert (out / "probe_convergence.environment.json").is_file()
    assert (out / "README.md").read_text(encoding="utf-8") == script.pc.render_readme(doc)


def test_a_published_proba_that_differs_by_one_ulp_fails_loudly_with_exit_1(run, capsys):
    script, argv, units, out = run
    path = units / "public@224/probas.npz"
    stored = dict(np.load(path))
    stored["p0"] = np.nextafter(stored["p0"], np.float32(2))
    np.savez(path, **stored)
    assert script.main(argv) == 1
    assert "SANITY" in capsys.readouterr().err
    doc = json.loads((out / "probe_convergence.json").read_text(encoding="utf-8"))
    assert doc["sanity"]["units"]["public@224"]["probas"]["differing_cells"] == ["public@224|cls|none|id"]


def test_the_output_inside_the_published_run_is_refused(run):
    script, argv, units, _ = run
    argv = [a for a in argv if not a.startswith("--out-dir")] + [f"--out-dir={units / 'x'}"]
    with pytest.raises(SystemExit):
        script.main(argv)
