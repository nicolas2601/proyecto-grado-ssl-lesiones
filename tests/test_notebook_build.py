"""The generated notebook must compile, execute its core, and stay in sync.

These tests are the reason we can trust a Kaggle run before spending a single
GPU-minute on it: a syntax error, a missing helper or a stale build all fail here
instead of 40 minutes into a queued session.
"""

import ast
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
from notebook_code_v5 import REPORT_CODE  # noqa: E402

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
    "stratified_summary",
    "fairness_gate",
    "estimate_fst",
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
        assert f"insertado desde src/dermssl/{name}.py" in core


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


def test_generated_notebook_has_an_empty_run_token_by_default():
    """Staging replaces this marker so result JSON can prove its freshness."""
    nb = builder.build_notebook()
    config = "".join(code_cells(nb)[0]["source"])
    report = "".join(code_cells(nb)[-1]["source"])
    assert 'KAGGLE_RUN_TOKEN = ""' in config
    assert '"kaggle_run_token": KAGGLE_RUN_TOKEN' in report


def test_no_credentials_are_baked_into_the_notebook():
    """A leaked Kaggle key in a committed notebook would be a real incident."""
    text = NOTEBOOK.read_text(encoding="utf-8")
    for forbidden in ("KAGGLE_KEY =", "KAGGLE_KEY=", "kaggle.json\":", "KGAT"):
        assert forbidden not in text, f"possible credential leak: {forbidden}"


def test_validate_detects_a_broken_cell():
    """Negative control: the validator must actually be able to fail."""
    broken = {"cells": [{"cell_type": "code", "id": "x", "source": ["def (:\n"]}]}
    assert builder.validate(broken) != []


def _fst_manifest_guard(tree: ast.Module) -> ast.If | None:
    """The `if FST_MANIFEST_PATH.is_file(): ...` statement in REPORT_CODE."""
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.If) and "FST_MANIFEST_PATH" in ast.dump(node.test)
        ),
        None,
    )


def test_report_code_guards_fst_manifest_path_before_touching_it():
    """REPORT_CODE must check for fst_manifest.csv before reading it."""
    tree = ast.parse(REPORT_CODE)
    guard = _fst_manifest_guard(tree)
    assert guard is not None, "expected an `if ...FST_MANIFEST_PATH...:` guard in REPORT_CODE"


def test_report_code_no_manifest_path_never_touches_fairness_gate():
    """Task 2.7: with no fst_manifest.csv, the run must stay byte-identical to
    v5 (design.md's error table: "Manifest missing -> ... v5 behavior
    unchanged"). Enforced structurally: every reference to `fairness_gate` or
    `stratified_summary` must live strictly inside the manifest-exists guard,
    so the no-manifest branch executes none of this PR's new code at all -
    gate_list, the printed report and RESULTS_SUMMARY_v5.json are therefore
    produced by exactly the same statements as before this change.
    """
    tree = ast.parse(REPORT_CODE)
    guard = _fst_manifest_guard(tree)
    assert guard is not None

    outside_dump = "\n".join(ast.dump(node) for node in tree.body if node is not guard)
    assert "fairness_gate" not in outside_dump
    assert "stratified_summary" not in outside_dump

    inside_dump = "\n".join(ast.dump(node) for node in guard.body)
    assert "fairness_gate" in inside_dump, "the guard must call fairness_gate when the manifest exists"


def test_fairness_path_uses_balanced_linear_probe_for_macro_f1():
    guard = _fst_manifest_guard(ast.parse(REPORT_CODE))
    assert guard is not None
    source = ast.get_source_segment(REPORT_CODE, guard)

    assert 'run_probe("logreg_balanced"' in source
    assert 'run_probe("knn"' not in source


def test_fairness_path_preserves_unavailable_and_missing_manifest_assignments():
    guard = _fst_manifest_guard(ast.parse(REPORT_CODE))
    assert guard is not None
    source = ast.get_source_segment(REPORT_CODE, guard)

    assert "fst_bucket_or_none" in source
    assert "fst_by_id.get(img_id" in source
    assert "if img_id in fst_by_id.index" not in source
