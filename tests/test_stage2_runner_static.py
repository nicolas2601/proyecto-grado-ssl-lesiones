"""Static contract for the GPU-only Stage 2 executable runner."""

import ast
import importlib.util
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "kaggle_circle_stage2.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("kaggle_circle_stage2", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage2_runner_exists_and_compiles():
    assert RUNNER.is_file()
    py_compile.compile(str(RUNNER), doraise=True)


def test_stage2_runner_imports_torch_only_after_argument_validation():
    source = RUNNER.read_text(encoding="utf-8")
    before_run = source.split("def run(")[0]

    assert "import torch" not in before_run
    assert "from dermssl.finetune import circle_torch" not in before_run


def test_stage2_runner_assembles_all_three_real_variants_and_reporting():
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        "ProbeHead",
        "build_probe_optimizer",
        "joint_regularization_loss",
        "synthetic_blend_mask",
        "stratified_summary",
        "comparison_report",
    ):
        assert required in source
    assert "knn" not in source.lower()


def test_stage2_runner_smoke_mode_uses_real_bundle_and_marks_results_non_final():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    source = RUNNER.read_text(encoding="utf-8")

    assert "--feature-bundle" in source
    assert "--stage1-gate" in source
    assert "--smoke" in source
    assert '"valid_for_thesis": not args.smoke' in source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        for node in ast.walk(tree)
    )


def test_stage2_runner_reads_the_gate_schema_written_by_v5(tmp_path):
    runner = _load_runner_module()
    gate_path = tmp_path / "RESULTS_SUMMARY_v5.json"
    gate_path.write_text(
        json.dumps(
            {
                "gates": [
                    {
                        "gate": "G3 equidad por fototipo",
                        "criterio": "gap <= 0.15",
                        "valor": 0.2,
                        "status": "FAIL",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert runner.load_stage1_gate_status(gate_path) == "FAIL"


def test_stage2_runner_uses_the_same_initialization_seed_for_all_variants():
    source = RUNNER.read_text(encoding="utf-8")

    assert "seed=args.seed + offset" not in source
    assert "seed=args.seed," in source
