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


def test_smoke_budget_is_small():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pretrain_script", ROOT / "scripts" / "pretrain.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SMOKE_BUDGET_GPU_HOURS <= 0.1
