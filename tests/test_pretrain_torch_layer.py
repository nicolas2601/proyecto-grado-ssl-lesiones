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


def test_dino_head_survives_deepcopy():
    """DinoContinual.__init__ builds the teacher via copy.deepcopy(student_head)
    (torch_dino.py). The legacy nn.utils.weight_norm hook-based
    implementation cannot be deepcopy'd on recent PyTorch (RuntimeError,
    "Only Tensors created explicitly by the user ... support the deepcopy
    protocol" - see https://github.com/pytorch/pytorch/pull/103001); this
    only reproduces where torch is actually installed (the GPU cluster),
    so it skips everywhere else instead of failing to import."""
    torch = pytest.importorskip("torch")
    import copy
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from dermssl.pretrain.torch_dino import DinoHead

    head = DinoHead(in_dim=384, out_dim=64)
    copy.deepcopy(head)  # must not raise


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


def test_imagenet_resnet_control_uses_pretrained_weights_in_the_ssl_builder():
    """The named ImageNet control must load ImageNet weights, not silently
    reuse the random SimSiam arm."""
    source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_loop.py").read_text()
    model_source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_simsiam.py").read_text()
    assert '"simsiam_imagenet"' in source
    assert "ResNet50_Weights.IMAGENET1K_V2" in model_source


def test_training_writes_the_run_contract_and_honors_an_explicit_resume_path():
    source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_loop.py").read_text()
    assert "run_contract.build_run_manifest" in source
    assert "cfg.resume_checkpoint" in source


def test_original_transformed_mode_uses_paired_images_and_named_consistency_loss():
    """The adaptation objective must be explicit: original/transform pairs go
    through a representation-consistency loss, not two arbitrary augmentations."""
    loop_source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_loop.py").read_text()
    data_source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_data.py").read_text()
    model_source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_simsiam.py").read_text()

    assert "collect_paired_images" in loop_source
    assert "original_transformed_consistency_loss" in loop_source
    assert "transformed_image_root" in loop_source
    assert "tuple[Path, Path]" in data_source
    assert "original_transformed_consistency_loss" in model_source


def test_smoke_run_limits_steps_and_accepts_the_transformed_image_root():
    script_source = (ROOT / "scripts" / "pretrain.py").read_text()
    loop_source = (ROOT / "src" / "dermssl" / "pretrain" / "torch_loop.py").read_text()

    assert "--transformed-image-root" in script_source
    assert "SMOKE_MAX_STEPS" in script_source
    assert "max_steps" in script_source
    assert "cfg.max_steps" in loop_source

