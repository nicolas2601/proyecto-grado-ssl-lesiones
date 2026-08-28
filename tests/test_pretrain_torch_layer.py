"""The torch layer cannot run here (no torch, no GPU) but it CAN be checked:
it must compile, it must not leak into the pure package, and the launch
script must fail on a bad config BEFORE importing torch."""

import ast
import py_compile
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TORCH_MODULES = sorted((ROOT / "src" / "dermssl" / "pretrain").glob("torch_*.py"))


def test_the_torch_modules_exist():
    names = [p.name for p in TORCH_MODULES]
    assert names == [
        "torch_data.py",
        "torch_dino.py",
        "torch_loop.py",
        "torch_simsiam.py",
    ]


@pytest.mark.parametrize("path", TORCH_MODULES, ids=lambda p: p.name)
def test_torch_modules_compile(path):
    """The cheapest possible proof the cluster will not die on a SyntaxError."""
    py_compile.compile(str(path), doraise=True)


def test_launch_script_compiles():
    py_compile.compile(str(ROOT / "scripts" / "pretrain.py"), doraise=True)


def test_importing_the_pure_package_never_imports_torch():
    """The v5 selling point stands in v6: the unit suite needs no GPU stack."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            del sys.modules[name]
    import dermssl.pretrain  # noqa: F401

    assert "torch" not in sys.modules


def test_pure_modules_do_not_mention_torch_in_imports():
    pure = [
        p
        for p in (ROOT / "src" / "dermssl" / "pretrain").glob("*.py")
        if not p.name.startswith("torch_")
    ]
    assert len(pure) >= 6  # __init__ + the tested core
    for path in pure:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("torch") for a in node.names), path.name
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("torch"), path.name


def test_launch_script_validates_config_before_importing_torch():
    """Order in main(): validate(cfg) must run before the torch_loop import,
    so a bad flag fails in milliseconds, not after a cluster queue wait."""
    source = (ROOT / "scripts" / "pretrain.py").read_text()
    main_body = source.split("def main(")[1]
    assert main_body.index("validate(cfg)") < main_body.index("torch_loop")


def test_torch_loop_gates_dino_paths_by_family_not_exact_method():
    """dino_scratch must inherit the EMA update and multi-crop path; a literal
    == "dino_continual" gate would silently skip the teacher update."""
    source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_loop.py").read_text()
    assert 'if cfg.method == "dino_continual"' not in source


def test_dino_model_exposes_a_pretrained_switch():
    """The scratch arm reuses DinoContinual with pretrained=False; the switch
    must exist in the constructor signature."""
    tree = ast.parse(
        (ROOT / "src" / "dermssl" / "pretrain" / "torch_dino.py").read_text()
    )
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "DinoContinual"
    )
    init = next(
        n
        for n in ast.walk(cls)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    assert "pretrained" in [a.arg for a in init.args.args + init.args.kwonlyargs]


def test_simsiam_model_exposes_backbone_and_pretrained_switches():
    """simsiam_dinov2 reuses the SimSiam class with a DINOv2 backbone; the
    constructor must accept the backbone name and a pretrained switch."""
    tree = ast.parse(
        (ROOT / "src" / "dermssl" / "pretrain" / "torch_simsiam.py").read_text()
    )
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "SimSiam"
    )
    init = next(
        n
        for n in ast.walk(cls)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    args = [a.arg for a in init.args.args + init.args.kwonlyargs]
    assert "backbone_name" in args
    assert "pretrained" in args


def test_torch_loop_gates_simsiam_paths_by_family():
    """simsiam_dinov2 must inherit the pair-dataset and loss path; literal
    == "simsiam" gates would route it into the DINO branch instead."""
    source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_loop.py").read_text()
    assert "SIMSIAM_METHODS" in source
    assert 'if cfg.method == "simsiam":' not in source


def test_smoke_budget_is_small():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pretrain_script", ROOT / "scripts" / "pretrain.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SMOKE_BUDGET_GPU_HOURS <= 0.1
