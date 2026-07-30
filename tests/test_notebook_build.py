"""The generated notebook must compile, execute its core, and stay in sync.

These tests are the reason we can trust a Kaggle run before spending a single
GPU-minute on it: a syntax error, a missing helper or a stale build all fail here
instead of 40 minutes into a queued session.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
NOTEBOOK = REPO / "notebooks" / "proyecto_ssl_lesiones_v5.ipynb"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_notebook_v5 as builder  # noqa: E402

# Every helper the torch-dependent cells call by name. If the inlined core stops
# providing one of these, the Kaggle run dies with NameError - so assert it here.
REQUIRED_API = [
    "resolve_datasets",
    "resolve_resolution",
    "patch_size_for",
    "stratified_regime_indices",
    "build_regimes",
    "describe_regimes",
    "summarize",
    "run_probe",
    "headline_metric",
    "PRIOR_DOMINATED_PROBES",
    "UNCALIBRATED_PROBES",
    "FeatureCache",
    "cache_key",
    "evaluate_gates",
    "report",
    "all_blocking_pass",
]


@pytest.fixture(scope="module")
def notebook():
    return builder.build_notebook()


def code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def test_every_code_cell_compiles(notebook):
    assert builder.validate(notebook) == []


def test_notebook_has_the_expected_shape(notebook):
    assert notebook["nbformat"] == 4
    assert len(code_cells(notebook)) >= 7
    assert notebook["cells"][0]["cell_type"] == "markdown"
    ids = [c["id"] for c in notebook["cells"]]
    assert len(set(ids)) == len(ids), "cell ids must be unique"


def test_core_cell_executes_and_exposes_the_full_api(notebook):
    """Executing the inlined core proves it runs, not merely that it parses."""
    core = "".join(code_cells(notebook)[1]["source"])
    namespace: dict = {"__name__": "__notebook_core__"}
    exec(compile(core, "<core cell>", "exec"), namespace)  # noqa: S102 - that is the test
    missing = [name for name in REQUIRED_API if name not in namespace]
    assert missing == [], f"inlined core is missing: {missing}"


def test_core_cell_has_no_future_import(notebook):
    """String annotations + @dataclass + exec do not mix (see module_source()).

    Checked as a statement, not as a substring: the cell legitimately *mentions*
    the import inside an explanatory comment.
    """
    core = "".join(code_cells(notebook)[1]["source"])
    statements = [ln.strip() for ln in core.splitlines() if not ln.strip().startswith("#")]
    assert "from __future__ import annotations" not in statements


def test_inlined_dataclasses_are_usable_after_exec(notebook):
    """Regression guard: @dataclass must construct inside the notebook namespace."""
    core = "".join(code_cells(notebook)[1]["source"])
    namespace: dict = {"__name__": "__notebook_core__"}
    exec(compile(core, "<core cell>", "exec"), namespace)  # noqa: S102
    resolved = namespace["ResolvedData"](backend="local")
    assert resolved.summary()["backend"] == "local"
    assert resolved.ham_images == {}


def test_core_cell_inlines_every_module(notebook):
    core = "".join(code_cells(notebook)[1]["source"])
    for name in builder.MODULE_ORDER:
        assert f"inlined from src/dermssl/{name}.py" in core


def test_committed_notebook_is_not_stale(notebook):
    """Guards against hand-edits and against forgetting to regenerate."""
    assert NOTEBOOK.is_file(), "run: python scripts/build_notebook_v5.py"
    assert NOTEBOOK.read_text(encoding="utf-8") == builder.serialize(notebook), (
        "notebook is out of date; run: python scripts/build_notebook_v5.py"
    )


def test_committed_notebook_is_valid_json_with_no_stored_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "commit notebooks without outputs"
            assert cell["execution_count"] is None


def test_smoke_is_enabled_by_default_in_the_committed_notebook():
    """A fresh Run All must be the cheap validation, never an 11-hour surprise."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    config = "".join(nb["cells"][1]["source"])
    assert "SMOKE        = True" in config


def test_no_credentials_are_baked_into_the_notebook():
    """A leaked Kaggle key in a committed notebook would be a real incident."""
    text = NOTEBOOK.read_text(encoding="utf-8")
    for forbidden in ("KAGGLE_KEY =", "KAGGLE_KEY=", "kaggle.json\":", "KGAT"):
        assert forbidden not in text, f"possible credential leak: {forbidden}"


def test_validate_detects_a_broken_cell():
    """Negative control: the validator must actually be able to fail."""
    broken = {"cells": [{"cell_type": "code", "id": "x", "source": ["def (:\n"]}]}
    assert builder.validate(broken) != []
